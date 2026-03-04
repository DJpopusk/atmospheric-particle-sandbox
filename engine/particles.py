import json
from pathlib import Path

import numpy as np

from engine.physics import (
    apply_atmospheric_forces,
    collide_domain_aabb,
    collide_obstacle_aabb,
    collide_particles_hashed,
)


def _build_uv_sphere(segments_u=14, segments_v=10):
    verts = []
    norms = []
    for iy in range(segments_v):
        v0 = iy / segments_v
        v1 = (iy + 1) / segments_v
        phi0 = np.pi * (v0 - 0.5)
        phi1 = np.pi * (v1 - 0.5)
        for ix in range(segments_u):
            u0 = ix / segments_u
            u1 = (ix + 1) / segments_u
            th0 = 2.0 * np.pi * u0
            th1 = 2.0 * np.pi * u1

            p00 = np.array([np.cos(phi0) * np.cos(th0), np.sin(phi0), np.cos(phi0) * np.sin(th0)], dtype=np.float32)
            p10 = np.array([np.cos(phi0) * np.cos(th1), np.sin(phi0), np.cos(phi0) * np.sin(th1)], dtype=np.float32)
            p01 = np.array([np.cos(phi1) * np.cos(th0), np.sin(phi1), np.cos(phi1) * np.sin(th0)], dtype=np.float32)
            p11 = np.array([np.cos(phi1) * np.cos(th1), np.sin(phi1), np.cos(phi1) * np.sin(th1)], dtype=np.float32)

            tris = (p00, p01, p11, p00, p11, p10)
            for p in tris:
                verts.append(p)
                norms.append(p / max(np.linalg.norm(p), 1e-6))

    return np.array(verts, dtype=np.float32), np.array(norms, dtype=np.float32)


def _read_particle_config(config_path: Path):
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    items = raw["particles"] if isinstance(raw, dict) and "particles" in raw else raw
    if not isinstance(items, list):
        raise ValueError("Particle config must be a list or a dict with 'particles'")
    particles = []
    for i, p in enumerate(items):
        if not isinstance(p, dict):
            raise ValueError(f"Particle #{i} is not an object")
        if "position" in p:
            position = p["position"]
        else:
            position = [p.get("x"), p.get("y"), p.get("z")]
        if "velocity" in p:
            velocity = p["velocity"]
        else:
            velocity = [p.get("vx", 0.0), p.get("vy", 0.0), p.get("vz", 0.0)]
        if len(position) != 3 or len(velocity) != 3:
            raise ValueError(f"Particle #{i} has invalid position/velocity dimensions")

        particles.append(
            {
                "position": np.array(position, dtype=np.float32),
                "velocity": np.array(velocity, dtype=np.float32),
                "mass": float(p.get("mass", 1.0)),
                "radius": float(p.get("radius", 0.08)),
            }
        )
    return particles


class ParticleSystem:
    def __init__(
        self,
        ctx,
        program,
        count=400,
        radius=0.08,
        bounds=None,
        obstacle_bounds=None,
        base_wind=(1.2, 0.08, 0.0),
        drag_coeff=0.9,
        turbulence=0.7,
        buoyancy=0.25,
        gravity=0.10,
        boundary_restitution=0.55,
        obstacle_restitution=0.25,
        obstacle_tangent_damping=0.85,
        particle_restitution=0.75,
        kick_strength=2.3,
        config_path=None,
    ):
        self.ctx = ctx
        self.program = program
        self.default_count = int(count)
        self.default_radius = float(radius)
        self.domain_bounds = bounds if bounds is not None else np.array([[-4, -2, -4], [4, 2, 4]], dtype=np.float32)
        self.obstacle_bounds = obstacle_bounds
        self.base_wind = np.array(base_wind, dtype=np.float32)
        self.drag_coeff = float(drag_coeff)
        self.turbulence = float(turbulence)
        self.buoyancy = float(buoyancy)
        self.gravity = float(gravity)
        self.boundary_restitution = float(boundary_restitution)
        self.obstacle_restitution = float(obstacle_restitution)
        self.obstacle_tangent_damping = float(obstacle_tangent_damping)
        self.particle_restitution = float(particle_restitution)
        self.kick_strength = float(kick_strength)
        self.time_s = 0.0

        self.config_particles = None
        if config_path is not None:
            cfg_path = Path(config_path)
            if cfg_path.exists():
                try:
                    self.config_particles = _read_particle_config(cfg_path)
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed to read particle config {cfg_path}: {exc}")
            else:
                print(f"Particle config not found: {cfg_path}")

        self.sphere_vbo = None
        self.sphere_nbo = None
        self.inst_cr_vbo = None
        self.inst_speed_vbo = None
        self.vao = None

        self.count = 0
        self.pos = np.zeros((0, 3), dtype=np.float32)
        self.vel = np.zeros((0, 3), dtype=np.float32)
        self.mass = np.zeros((0,), dtype=np.float32)
        self.radius = np.zeros((0,), dtype=np.float32)

        self._create_sphere_mesh()
        self.reset(bounds=self.domain_bounds)

    def _create_sphere_mesh(self):
        verts, norms = _build_uv_sphere()
        self.sphere_vbo = self.ctx.buffer(verts.tobytes())
        self.sphere_nbo = self.ctx.buffer(norms.tobytes())

    def _rebuild_instances(self, new_count: int):
        self.count = int(new_count)
        self.pos = np.zeros((self.count, 3), dtype=np.float32)
        self.vel = np.zeros((self.count, 3), dtype=np.float32)
        self.mass = np.ones((self.count,), dtype=np.float32)
        self.radius = np.full((self.count,), self.default_radius, dtype=np.float32)

        if self.inst_cr_vbo is not None:
            self.inst_cr_vbo.release()
        if self.inst_speed_vbo is not None:
            self.inst_speed_vbo.release()

        self.inst_cr_vbo = self.ctx.buffer(reserve=self.count * 4 * 4)
        self.inst_speed_vbo = self.ctx.buffer(reserve=self.count * 4)
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.sphere_vbo, "3f", "in_pos"),
                (self.sphere_nbo, "3f", "in_norm"),
                (self.inst_cr_vbo, "4f/i", "in_center_radius"),
                (self.inst_speed_vbo, "1f/i", "in_speed"),
            ],
        )

    def _sync_instance_buffers(self):
        center_radius = np.hstack([self.pos, self.radius[:, None]]).astype(np.float32)
        speed = np.linalg.norm(self.vel, axis=1).astype(np.float32)
        self.inst_cr_vbo.write(center_radius.tobytes())
        self.inst_speed_vbo.write(speed.tobytes())

    def set_uniform_radius(self, r: float):
        self.radius[:] = float(np.clip(r, 0.01, 0.50))
        self._sync_instance_buffers()

    def reset(self, bounds):
        self.domain_bounds = np.array(bounds, dtype=np.float32)
        lo, hi = self.domain_bounds[0], self.domain_bounds[1]
        span = hi - lo

        if self.config_particles is not None and len(self.config_particles) > 0:
            if self.count != len(self.config_particles):
                self._rebuild_instances(len(self.config_particles))
            for i, p in enumerate(self.config_particles):
                self.pos[i] = p["position"]
                self.vel[i] = p["velocity"]
                self.mass[i] = max(1e-3, p["mass"])
                self.radius[i] = max(0.01, p["radius"])
        else:
            if self.count != self.default_count:
                self._rebuild_instances(self.default_count)
            self.pos = lo + np.random.rand(self.count, 3).astype(np.float32) * span
            self.vel = (np.random.rand(self.count, 3).astype(np.float32) - 0.5) * 0.35
            self.mass = 0.8 + 0.4 * np.random.rand(self.count).astype(np.float32)
            self.radius = self.default_radius * (0.8 + 0.4 * np.random.rand(self.count).astype(np.float32))

        if self.obstacle_bounds is not None:
            obs = np.array(self.obstacle_bounds, dtype=np.float32)
            for _ in range(10):
                inside = np.logical_and(self.pos >= obs[0], self.pos <= obs[1]).all(axis=1)
                if not inside.any():
                    break
                self.pos[inside] = lo + np.random.rand(int(inside.sum()), 3).astype(np.float32) * span

        self.time_s = 0.0
        self._sync_instance_buffers()

    def kick(self, direction=None, strength=None):
        if direction is None:
            direction = np.array([1.0, 0.2, 0.0], dtype=np.float32)
        if strength is None:
            strength = self.kick_strength
        direction = np.array(direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        noise = (np.random.rand(self.count, 3).astype(np.float32) - 0.5) * 0.35
        self.vel += direction * (float(strength) / np.sqrt(np.maximum(self.mass, 1e-6)))[:, None] + noise

    def impulse_at_point(self, point, direction=None, strength=None, radius=1.0):
        if strength is None:
            strength = self.kick_strength * 1.2
        p = np.array(point, dtype=np.float32)
        d = self.pos - p[None, :]
        dist = np.linalg.norm(d, axis=1)
        mask = dist < radius
        if not mask.any():
            return
        n = d[mask] / np.maximum(dist[mask][:, None], 1e-5)
        if direction is not None:
            dir_vec = np.array(direction, dtype=np.float32)
            dir_vec = dir_vec / max(float(np.linalg.norm(dir_vec)), 1e-6)
            n = n * 0.65 + dir_vec[None, :] * 0.35
            n = n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-6)
        w = (1.0 - dist[mask] / max(radius, 1e-5)) ** 2
        self.vel[mask] += n * (float(strength) * w)[:, None] / np.sqrt(np.maximum(self.mass[mask], 1e-6))[:, None]

    def step(self, dt: float):
        apply_atmospheric_forces(
            self.pos,
            self.vel,
            self.mass,
            dt,
            self.time_s,
            base_wind=self.base_wind,
            drag_coeff=self.drag_coeff,
            turbulence=self.turbulence,
            buoyancy=self.buoyancy,
            gravity=self.gravity,
        )

        collide_particles_hashed(
            self.pos,
            self.vel,
            self.radius,
            self.mass,
            restitution=self.particle_restitution,
        )

        self.pos += self.vel * dt

        collide_domain_aabb(
            self.pos,
            self.vel,
            self.domain_bounds,
            self.radius,
            restitution=self.boundary_restitution,
        )
        if self.obstacle_bounds is not None:
            collide_obstacle_aabb(
                self.pos,
                self.vel,
                self.obstacle_bounds,
                self.radius,
                restitution=self.obstacle_restitution,
                tangent_damping=self.obstacle_tangent_damping,
            )

        self.vel *= 0.9993
        self.time_s += dt
        self._sync_instance_buffers()

    def draw(self, proj, view, light_dir):
        mvp = np.array(view @ proj, dtype=np.float32).T
        self.program["mvp"].write(mvp.tobytes())
        view3 = np.array(view, dtype=np.float32)[:3, :3]
        self.program["normal_matrix"].write(np.linalg.inv(view3).T.astype("f4").tobytes())
        self.program["light_dir"].value = tuple(light_dir)
        self.vao.render(mode=self.ctx.TRIANGLES, instances=self.count)
