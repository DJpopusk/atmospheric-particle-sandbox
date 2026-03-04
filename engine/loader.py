import pathlib
import shutil
import subprocess
import tempfile
import logging
from dataclasses import dataclass

import moderngl
import numpy as np
import trimesh

logging.getLogger("trimesh").setLevel(logging.ERROR)


@dataclass
class Model:
    vao: moderngl.VertexArray
    vertex_count: int
    aabb: np.ndarray  # shape (2,3): min, max
    size: np.ndarray
    center: np.ndarray
    label: str
    is_fallback: bool

    def draw(self, proj, view, light_dir):
        # Pyrr matrices are row-major (row-vector math). Convert for GLSL column-vector use.
        mvp = np.array(view @ proj, dtype='f4').T
        self.vao.program['mvp'].write(mvp.tobytes())
        view3 = np.array(view, dtype='f4')[:3, :3]
        self.vao.program['normal_matrix'].write(np.linalg.inv(view3).T.astype('f4').tobytes())
        self.vao.program['light_dir'].value = tuple(light_dir)
        self.vao.render(mode=moderngl.TRIANGLES)


def _fallback_cube():
    # Simple unit cube centered at origin
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float32)
    normals = np.array([
        [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
        [0, 0,  1], [0, 0,  1], [0, 0,  1], [0, 0,  1],
    ], dtype=np.float32)
    idx = np.array([
        0, 1, 2, 2, 3, 0,  # back
        4, 5, 6, 6, 7, 4,  # front
        0, 4, 7, 7, 3, 0,  # left
        1, 5, 6, 6, 2, 1,  # right
        3, 2, 6, 6, 7, 3,  # top
        0, 1, 5, 5, 4, 0,  # bottom
    ], dtype=np.int32)
    return vertices[idx], normals[idx]


def _load_mesh_any(path: pathlib.Path):
    primary_error = None
    try:
        loaded = trimesh.load(path)
    except Exception as exc:  # noqa: BLE001
        primary_error = str(exc)
        loaded = None

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if isinstance(loaded, trimesh.Scene):
        if loaded.is_empty or len(loaded.geometry) == 0:
            loaded = None
        else:
            try:
                return loaded.dump(concatenate=True)
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to merge scene meshes from {path}. Reason: {exc}")
                loaded = None

    if loaded is None:
        assimp = shutil.which("assimp")
        if assimp is None:
            if primary_error is not None:
                print(f"Primary trimesh load failed for {path}. Reason: {primary_error}")
            return None
        try:
            with tempfile.TemporaryDirectory(prefix="particle_sandbox_") as tmp:
                converted = pathlib.Path(tmp) / "converted.obj"
                run = subprocess.run(
                    [assimp, "export", str(path), str(converted), "-fobj"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if run.returncode != 0 or not converted.exists():
                    return None
                loaded_assimp = trimesh.load(converted)
                if isinstance(loaded_assimp, trimesh.Trimesh):
                    return loaded_assimp
                if isinstance(loaded_assimp, trimesh.Scene):
                    if loaded_assimp.is_empty or len(loaded_assimp.geometry) == 0:
                        return None
                    return loaded_assimp.dump(concatenate=True)
        except Exception as exc:  # noqa: BLE001
            print(f"Assimp conversion failed for {path}. Reason: {exc}")
            return None
        if primary_error is not None:
            print(f"Primary trimesh load failed for {path}. Reason: {primary_error}")
    return None


def load_model(ctx: moderngl.Context, program: moderngl.Program, path: pathlib.Path) -> Model:
    mesh = None
    model_label = path.name
    if path.exists():
        mesh = _load_mesh_any(path)
        if mesh is None:
            print(f"Model {path} could not be parsed as mesh/scene, using fallback cube.")
    else:
        print(f"Model file {path} not found, using fallback cube.")

    if mesh is None or mesh.is_empty:
        model_label = "fallback_cube"
        verts, norms = _fallback_cube()
        aabb = np.array([verts.min(axis=0), verts.max(axis=0)], dtype=np.float32)
        is_fallback = True
    else:
        tris = np.array(mesh.triangles, dtype=np.float32)  # (T,3,3)
        verts = tris.reshape(-1, 3)
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
            # Map vertex normals per triangle vertex if available.
            faces = np.array(mesh.faces, dtype=np.int32)
            norms = np.array(mesh.vertex_normals, dtype=np.float32)[faces].reshape(-1, 3)
        else:
            # Compute flat normals per triangle.
            e1 = tris[:, 1] - tris[:, 0]
            e2 = tris[:, 2] - tris[:, 0]
            fn = np.cross(e1, e2)
            fn_len = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
            fn = fn / fn_len
            norms = np.repeat(fn, 3, axis=0)
        bounds = mesh.bounds
        aabb = np.array([bounds[0], bounds[1]], dtype=np.float32)
        is_fallback = False

    # recenter geometry around origin so camera & particles share the same frame
    center = (aabb[0] + aabb[1]) * 0.5
    verts = verts - center
    aabb = aabb - center
    center = np.zeros(3, dtype=np.float32)
    size = aabb[1] - aabb[0]

    vbo = ctx.buffer(verts.tobytes())
    nbo = ctx.buffer(norms.tobytes())
    vao = ctx.vertex_array(program, [(vbo, '3f', 'in_pos'), (nbo, '3f', 'in_norm')])
    return Model(
        vao=vao,
        vertex_count=len(verts),
        aabb=aabb,
        size=size,
        center=center,
        label=model_label,
        is_fallback=is_fallback,
    )
