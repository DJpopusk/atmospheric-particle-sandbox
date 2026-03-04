import argparse
import math
import pathlib
import sys

import moderngl
import numpy as np
from pyrr import Matrix44, Vector3

from engine.loader import load_model
from engine.particles import ParticleSystem
from engine.physics import run_atmosphere_self_check

_PRELOADED_IMGUI = None
_PRELOADED_IMGUI_RENDERER = None
_PRELOADED_IMGUI_ERROR = None
try:
    from imgui_bundle import imgui as _PRELOADED_IMGUI
    from imgui_bundle.python_backends.glfw_backend import GlfwRenderer as _PRELOADED_IMGUI_RENDERER
except Exception as exc:  # noqa: BLE001
    _PRELOADED_IMGUI_ERROR = str(exc)

import glfw

WIN_W, WIN_H = 1280, 720
BACKGROUND = (0.035, 0.055, 0.085)


def load_imgui_backend():
    if _PRELOADED_IMGUI is not None and _PRELOADED_IMGUI_RENDERER is not None:
        return _PRELOADED_IMGUI, _PRELOADED_IMGUI_RENDERER, None
    return None, None, _PRELOADED_IMGUI_ERROR


class OrbitCamera:
    def __init__(self, distance=10.0, yaw=-0.9, pitch=0.4, target=Vector3([0.0, 1.0, 0.0])):
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.target = target
        self._last = None
        self._drag_px = 0.0

    def handle_mouse(self, window, action, mods):
        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            self._last = (x, y)
            self._drag_px = 0.0
            return False
        elif action == glfw.RELEASE:
            was_click = self._last is not None and self._drag_px < 4.0
            self._last = None
            return was_click
        return False

    def drag(self, window):
        if self._last is None:
            return
        x, y = glfw.get_cursor_pos(window)
        lx, ly = self._last
        self._drag_px += math.hypot(x - lx, y - ly)
        self.orbit((x - lx) * 0.005, (y - ly) * 0.005)
        self._last = (x, y)

    def orbit(self, yaw_delta, pitch_delta):
        self.yaw += yaw_delta
        self.pitch = float(np.clip(self.pitch + pitch_delta, -1.2, 1.2))

    def scroll(self, offset):
        self.distance = float(np.clip(self.distance * (1.0 - offset * 0.1), 2.5, 40.0))

    def view_matrix(self):
        cx = self.target.x + self.distance * math.cos(self.pitch) * math.cos(self.yaw)
        cy = self.target.y + self.distance * math.sin(self.pitch)
        cz = self.target.z + self.distance * math.cos(self.pitch) * math.sin(self.yaw)
        eye = Vector3([cx, cy, cz])
        up = Vector3([0.0, 1.0, 0.0])
        return Matrix44.look_at(eye, self.target, up)

    def forward_vector(self):
        cx = self.target.x + self.distance * math.cos(self.pitch) * math.cos(self.yaw)
        cy = self.target.y + self.distance * math.sin(self.pitch)
        cz = self.target.z + self.distance * math.cos(self.pitch) * math.sin(self.yaw)
        forward = np.array([self.target.x - cx, self.target.y - cy, self.target.z - cz], dtype=np.float32)
        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return forward / norm


def _pick_ray_from_mouse(mouse_x, mouse_y, width, height, proj, view):
    x = (2.0 * mouse_x) / max(width, 1) - 1.0
    y = 1.0 - (2.0 * mouse_y) / max(height, 1)

    proj_col = np.array(proj, dtype=np.float32).T
    view_col = np.array(view, dtype=np.float32).T
    inv = np.linalg.inv(proj_col @ view_col)

    near_clip = np.array([x, y, -1.0, 1.0], dtype=np.float32)
    far_clip = np.array([x, y, 1.0, 1.0], dtype=np.float32)
    near_world = inv @ near_clip
    far_world = inv @ far_clip
    near_world = near_world[:3] / max(near_world[3], 1e-8)
    far_world = far_world[:3] / max(far_world[3], 1e-8)
    direction = far_world - near_world
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    return near_world.astype(np.float32), direction.astype(np.float32)


def _ray_aabb_hit(ray_origin, ray_dir, bounds):
    inv_dir = 1.0 / np.where(np.abs(ray_dir) < 1e-8, 1e-8, ray_dir)
    t1 = (bounds[0] - ray_origin) * inv_dir
    t2 = (bounds[1] - ray_origin) * inv_dir
    tmin = np.maximum.reduce(np.minimum(t1, t2))
    tmax = np.minimum.reduce(np.maximum(t1, t2))
    if tmax < 0.0 or tmin > tmax:
        return None
    t_hit = tmin if tmin >= 0.0 else tmax
    if t_hit < 0.0:
        return None
    return ray_origin + ray_dir * t_hit


def _build_aabb_lines(bounds, inflate=1.02):
    lo = np.array(bounds[0], dtype=np.float32)
    hi = np.array(bounds[1], dtype=np.float32)
    center = (lo + hi) * 0.5
    half = (hi - lo) * 0.5 * inflate
    lo = center - half
    hi = center + half
    corners = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    lines = []
    for a, b in edges:
        lines.append(corners[a])
        lines.append(corners[b])
    return np.array(lines, dtype=np.float32)


def _build_grid_lines(extent, step, y):
    coords = np.arange(-extent, extent + 0.5 * step, step, dtype=np.float32)
    lines = []
    for c in coords:
        lines.append(np.array([c, y, -extent], dtype=np.float32))
        lines.append(np.array([c, y, extent], dtype=np.float32))
        lines.append(np.array([-extent, y, c], dtype=np.float32))
        lines.append(np.array([extent, y, c], dtype=np.float32))
    return np.array(lines, dtype=np.float32)


def _build_axis_lines(length):
    axis_x = np.array([[-length, 0.0, 0.0], [length, 0.0, 0.0]], dtype=np.float32)
    axis_y = np.array([[0.0, -length, 0.0], [0.0, length, 0.0]], dtype=np.float32)
    axis_z = np.array([[0.0, 0.0, -length], [0.0, 0.0, length]], dtype=np.float32)
    return axis_x, axis_y, axis_z


class SceneGuides:
    def __init__(self, ctx, program, model_bounds, scene_extent):
        self.ctx = ctx
        self.program = program
        grid_extent = max(4.0, scene_extent * 2.0)
        grid_step = max(0.5, scene_extent / 6.0)
        grid_y = float(model_bounds[0][1] - scene_extent * 0.20)
        axis_len = max(1.5, scene_extent * 1.2)

        self.grid_vao = self._make_vao(_build_grid_lines(grid_extent, grid_step, grid_y))
        self.box_vao = self._make_vao(_build_aabb_lines(model_bounds, inflate=1.03))
        axis_x, axis_y, axis_z = _build_axis_lines(axis_len)
        self.axis_x_vao = self._make_vao(axis_x)
        self.axis_y_vao = self._make_vao(axis_y)
        self.axis_z_vao = self._make_vao(axis_z)

    def _make_vao(self, lines: np.ndarray):
        vbo = self.ctx.buffer(lines.astype(np.float32).tobytes())
        return self.ctx.vertex_array(self.program, [(vbo, "3f", "in_pos")])

    def _set_mvp(self, proj, view):
        mvp = np.array(view @ proj, dtype=np.float32).T
        self.program["mvp"].write(mvp.tobytes())

    def draw(self, proj, view):
        self._set_mvp(proj, view)

        # Grid: subtle depth-tested reference plane.
        self.program["color"].value = (0.16, 0.25, 0.34, 0.52)
        self.grid_vao.render(mode=moderngl.LINES)

        # Highlight obstacle bounds and world axes on top.
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.program["color"].value = (0.95, 0.86, 0.30, 1.0)
        self.box_vao.render(mode=moderngl.LINES)
        self.program["color"].value = (1.00, 0.30, 0.30, 1.0)
        self.axis_x_vao.render(mode=moderngl.LINES)
        self.program["color"].value = (0.30, 0.95, 0.45, 1.0)
        self.axis_y_vao.render(mode=moderngl.LINES)
        self.program["color"].value = (0.35, 0.60, 1.00, 1.0)
        self.axis_z_vao.render(mode=moderngl.LINES)
        self.ctx.enable(moderngl.DEPTH_TEST)


def init_window(width, height):
    if not glfw.init():
        raise SystemExit("Failed to init GLFW")
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # for macOS core profile
    window = glfw.create_window(width, height, "Atmospheric Flow Sandbox", None, None)
    if not window:
        glfw.terminate()
        raise SystemExit("Failed to create window")
    glfw.make_context_current(window)
    return window


def compile_programs(ctx: moderngl.Context):
    with open("shaders/model.vert", "r", encoding="ascii") as f:
        model_vs = f.read()
    with open("shaders/model.frag", "r", encoding="ascii") as f:
        model_fs = f.read()
    with open("shaders/particle.vert", "r", encoding="ascii") as f:
        p_vs = f.read()
    with open("shaders/particle.frag", "r", encoding="ascii") as f:
        p_fs = f.read()
    with open("shaders/line.vert", "r", encoding="ascii") as f:
        line_vs = f.read()
    with open("shaders/line.frag", "r", encoding="ascii") as f:
        line_fs = f.read()
    model_prog = ctx.program(vertex_shader=model_vs, fragment_shader=model_fs)
    particle_prog = ctx.program(vertex_shader=p_vs, fragment_shader=p_fs)
    line_prog = ctx.program(vertex_shader=line_vs, fragment_shader=line_fs)
    return model_prog, particle_prog, line_prog


class RuntimeControlPanel:
    def __init__(self, args):
        self.active = False
        self._gust_request = False
        self._reset_request = False
        self._mode = None
        self._web_url = None
        if self._init_tk(args):
            self._mode = "tk"
            self.active = True
            return
        if self._init_web(args):
            self._mode = "web"
            self.active = True
            return
        print("Control panel disabled")

    def _init_tk(self, args):
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:  # noqa: BLE001
            print(f"Tk control panel unavailable: {exc}")
            return False

        self._tk = tk
        self._ttk = ttk
        try:
            self.root = tk.Tk()
        except Exception as exc:  # noqa: BLE001
            print(f"Tk control panel disabled: {exc}")
            return False

        self.root.title("Flow Controls")
        self.root.geometry("360x690")
        self.root.resizable(False, True)
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(250, lambda: self.root.attributes("-topmost", False))
        self.vars = {}

        row = 0
        row = self._add_scale(row, "Wind X", "wind_x", -4.0, 4.0, 0.01, args.wind[0])
        row = self._add_scale(row, "Wind Y", "wind_y", -2.0, 2.0, 0.01, args.wind[1])
        row = self._add_scale(row, "Wind Z", "wind_z", -4.0, 4.0, 0.01, args.wind[2])
        row = self._add_scale(row, "Turbulence", "turbulence", 0.0, 3.0, 0.01, args.turbulence)
        row = self._add_scale(row, "Drag", "drag", 0.0, 3.0, 0.01, args.drag)
        row = self._add_scale(row, "Buoyancy", "buoyancy", 0.0, 2.0, 0.01, args.buoyancy)
        row = self._add_scale(row, "Gravity", "gravity", 0.0, 2.0, 0.01, args.gravity)
        row = self._add_scale(row, "Particle Size", "size", 0.02, 0.20, 0.005, args.size)
        row = self._add_scale(row, "Kick Strength", "kick_strength", 0.1, 8.0, 0.05, args.kick_strength)
        row = self._add_scale(row, "Max dt", "max_dt", 0.005, 0.05, 0.001, args.max_dt)

        self.pause_var = tk.BooleanVar(value=False)
        pause_box = ttk.Checkbutton(self.root, text="Pause", variable=self.pause_var)
        pause_box.grid(row=row, column=0, sticky="w", padx=10, pady=(6, 2))
        row += 1

        buttons = ttk.Frame(self.root)
        buttons.grid(row=row, column=0, sticky="ew", padx=8, pady=6)
        ttk.Button(buttons, text="Gust", command=self._request_gust).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Reset Particles", command=self._request_reset).grid(row=0, column=1, padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        return True

    def _init_web(self, args):
        try:
            import json
            import threading
            import webbrowser
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        except Exception as exc:  # noqa: BLE001
            print(f"Web control panel unavailable: {exc}")
            return False

        self._json = json
        self._threading = threading
        self._state_lock = threading.Lock()
        self._state = {
            "wind_x": float(args.wind[0]),
            "wind_y": float(args.wind[1]),
            "wind_z": float(args.wind[2]),
            "turbulence": float(args.turbulence),
            "drag": float(args.drag),
            "buoyancy": float(args.buoyancy),
            "gravity": float(args.gravity),
            "size": float(args.size),
            "kick_strength": float(args.kick_strength),
            "max_dt": float(args.max_dt),
            "pause": False,
        }
        panel = self
        html = self._build_web_html()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A003
                return

            def _send_json(self, payload, status=200):
                data = panel._json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_html(self, body):
                data = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                if self.path == "/":
                    self._send_html(html)
                    return
                if self.path == "/state":
                    with panel._state_lock:
                        payload = dict(panel._state)
                    self._send_json(payload, status=200)
                    return
                self._send_json({"error": "not found"}, status=404)

            def do_POST(self):  # noqa: N802
                if self.path == "/state":
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length)
                    try:
                        data = panel._json.loads(raw.decode("utf-8"))
                    except Exception:  # noqa: BLE001
                        self._send_json({"error": "bad json"}, status=400)
                        return
                    with panel._state_lock:
                        for key in panel._state:
                            if key in data:
                                panel._state[key] = data[key]
                    self._send_json({"ok": True}, status=200)
                    return
                if self.path == "/action/gust":
                    panel._request_gust()
                    self._send_json({"ok": True}, status=200)
                    return
                if self.path == "/action/reset":
                    panel._request_reset()
                    self._send_json({"ok": True}, status=200)
                    return
                self._send_json({"error": "not found"}, status=404)

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except Exception as exc:  # noqa: BLE001
            print(f"Web control panel failed to start: {exc}")
            return False
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        port = self._server.server_address[1]
        self._web_url = f"http://127.0.0.1:{port}/"
        print(f"Web control panel: {self._web_url}")
        try:
            webbrowser.open(self._web_url)
        except Exception:  # noqa: BLE001
            pass
        return True

    def _add_scale(self, row, label, key, low, high, step, initial):
        var = self._tk.DoubleVar(value=float(initial))
        scale = self._tk.Scale(
            self.root,
            from_=low,
            to=high,
            resolution=step,
            orient=self._tk.HORIZONTAL,
            label=label,
            variable=var,
            length=330,
        )
        scale.grid(row=row, column=0, sticky="ew", padx=8, pady=2)
        self.vars[key] = var
        return row + 1

    def _request_gust(self):
        self._gust_request = True

    def _request_reset(self):
        self._reset_request = True

    def _on_close(self):
        self.active = False
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass

    def _build_web_html(self):
        return """<!doctype html>
<html><head><meta charset="utf-8"><title>Flow Controls</title>
<style>
body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;margin:14px;background:#0f172a;color:#e2e8f0}
h2{margin:0 0 12px 0} .row{margin:8px 0} label{display:block;font-size:13px;margin-bottom:2px}
input[type=range]{width:100%} .val{float:right;color:#93c5fd} button{margin-right:6px;padding:6px 10px}
.box{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:12px;max-width:420px}
</style></head>
<body><div class="box"><h2>Flow Controls</h2>
<div id="sliders"></div>
<div class="row"><label><input id="pause" type="checkbox"> Pause</label></div>
<div class="row"><button id="gust">Gust</button><button id="reset">Reset Particles</button></div>
</div>
<script>
const specs=[
  ['wind_x','Wind X',-4,4,0.01],['wind_y','Wind Y',-2,2,0.01],['wind_z','Wind Z',-4,4,0.01],
  ['turbulence','Turbulence',0,3,0.01],['drag','Drag',0,3,0.01],['buoyancy','Buoyancy',0,2,0.01],
  ['gravity','Gravity',0,2,0.01],['size','Particle Size',0.02,0.2,0.005],['kick_strength','Kick Strength',0.1,8,0.05],
  ['max_dt','Max dt',0.005,0.05,0.001]
];
const slidersDiv=document.getElementById('sliders');
function mkSlider(k,l,min,max,step,val){
  const r=document.createElement('div');r.className='row';
  const lab=document.createElement('label');
  const span=document.createElement('span');span.className='val';span.id='v_'+k;span.textContent=Number(val).toFixed(3);
  lab.textContent=l;lab.appendChild(span);
  const s=document.createElement('input');s.type='range';s.min=min;s.max=max;s.step=step;s.value=val;s.id=k;
  s.oninput=()=>{span.textContent=Number(s.value).toFixed(3);push();};
  r.appendChild(lab);r.appendChild(s);slidersDiv.appendChild(r);
}
async function pull(){
  const r=await fetch('/state');const st=await r.json();
  specs.forEach(sp=>mkSlider(sp[0],sp[1],sp[2],sp[3],sp[4],st[sp[0]]));
  const p=document.getElementById('pause');p.checked=!!st.pause;p.onchange=push;
}
async function push(){
  const st={};specs.forEach(sp=>{st[sp[0]]=parseFloat(document.getElementById(sp[0]).value);});
  st.pause=document.getElementById('pause').checked;
  await fetch('/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(st)});
}
document.getElementById('gust').onclick=()=>fetch('/action/gust',{method:'POST'});
document.getElementById('reset').onclick=()=>fetch('/action/reset',{method:'POST'});
pull();
</script></body></html>"""

    def poll(self):
        if not self.active:
            return None
        if self._mode == "tk":
            try:
                self.root.update_idletasks()
                self.root.update()
            except self._tk.TclError:
                self.active = False
                return None
            values = {
                "wind": (
                    float(self.vars["wind_x"].get()),
                    float(self.vars["wind_y"].get()),
                    float(self.vars["wind_z"].get()),
                ),
                "turbulence": float(self.vars["turbulence"].get()),
                "drag": float(self.vars["drag"].get()),
                "buoyancy": float(self.vars["buoyancy"].get()),
                "gravity": float(self.vars["gravity"].get()),
                "size": float(self.vars["size"].get()),
                "kick_strength": float(self.vars["kick_strength"].get()),
                "max_dt": float(self.vars["max_dt"].get()),
                "pause": bool(self.pause_var.get()),
                "gust": self._gust_request,
                "reset": self._reset_request,
            }
        else:
            with self._state_lock:
                values = {
                    "wind": (
                        float(self._state["wind_x"]),
                        float(self._state["wind_y"]),
                        float(self._state["wind_z"]),
                    ),
                    "turbulence": float(self._state["turbulence"]),
                    "drag": float(self._state["drag"]),
                    "buoyancy": float(self._state["buoyancy"]),
                    "gravity": float(self._state["gravity"]),
                    "size": float(self._state["size"]),
                    "kick_strength": float(self._state["kick_strength"]),
                    "max_dt": float(self._state["max_dt"]),
                    "pause": bool(self._state["pause"]),
                    "gust": self._gust_request,
                    "reset": self._reset_request,
                }
        self._gust_request = False
        self._reset_request = False
        return values

    def close(self):
        if not self.active:
            return
        self.active = False
        if self._mode == "tk":
            self._on_close()
            return
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:  # noqa: BLE001
            pass


def main():
    parser = argparse.ArgumentParser(description="3D particle-object sandbox")
    parser.add_argument("--model", type=pathlib.Path, default=pathlib.Path("assets/model_cube.obj"),
                        help="Path to model file (.glb/.obj/.dae/.3ds..., default: assets/model_cube.obj)")
    parser.add_argument("--particle-config", type=pathlib.Path, default=None,
                        help="Optional JSON file with explicit particle list")
    parser.add_argument("--particles", type=int, default=400, help="Number of particles")
    parser.add_argument("--size", type=float, default=0.08, help="Particle radius")
    parser.add_argument("--wind", type=float, nargs=3, default=(1.2, 0.08, 0.0), metavar=("WX", "WY", "WZ"),
                        help="Base wind vector")
    parser.add_argument("--drag", type=float, default=0.9, help="Flow drag coefficient")
    parser.add_argument("--turbulence", type=float, default=0.7, help="Turbulence strength")
    parser.add_argument("--buoyancy", type=float, default=0.25, help="Buoyancy coefficient")
    parser.add_argument("--gravity", type=float, default=0.10, help="Effective gravity term")
    parser.add_argument("--kick-strength", type=float, default=2.3, help="Impulse strength for F/Enter/RMB")
    parser.add_argument("--max-dt", type=float, default=0.033, help="Max simulation timestep")
    parser.add_argument("--domain-scale", type=float, default=0.9, help="Domain padding scale from model size")
    parser.add_argument("--domain-min", type=float, nargs=3, default=(2.5, 2.0, 2.5),
                        metavar=("PX", "PY", "PZ"), help="Minimum domain padding")
    parser.add_argument("--boundary-restitution", type=float, default=0.55, help="Domain wall restitution")
    parser.add_argument("--obstacle-restitution", type=float, default=0.25, help="Obstacle restitution")
    parser.add_argument("--obstacle-tangent-damping", type=float, default=0.85,
                        help="Obstacle tangent damping")
    parser.add_argument("--no-imgui-panel", action="store_true",
                        help="Disable in-window ImGui controls")
    parser.add_argument("--no-control-panel", action="store_true",
                        help="Disable visual control panel window")
    parser.add_argument("--verify", action="store_true", help="Run headless physics self-check and exit")
    parser.add_argument("--verify-steps", type=int, default=480, help="Self-check integration steps")
    args = parser.parse_args()

    if args.verify:
        result = run_atmosphere_self_check(
            steps=args.verify_steps,
            count=max(200, args.particles),
            base_wind=args.wind,
            drag_coeff=args.drag,
            turbulence=args.turbulence,
            buoyancy=args.buoyancy,
            gravity=args.gravity,
            boundary_restitution=args.boundary_restitution,
            obstacle_restitution=args.obstacle_restitution,
            obstacle_tangent_damping=args.obstacle_tangent_damping,
            radius=args.size,
        )
        if not result["ok"]:
            raise SystemExit(f"Self-check failed: {result['reason']}")
        print(
            "Self-check OK: "
            f"particles={result['particles']} steps={result['steps']} "
            f"mean_speed={result['mean_speed']:.3f} inside_obstacle_ratio={result['inside_obstacle_ratio']:.4f}"
        )
        return

    window = init_window(WIN_W, WIN_H)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.BLEND)

    model_prog, particle_prog, line_prog = compile_programs(ctx)

    model = load_model(ctx, model_prog, args.model)
    model_title = "fallback cube" if model.is_fallback else model.label
    glfw.set_window_title(window, f"Atmospheric Flow Sandbox - {model_title}")
    padding = np.maximum(model.size * args.domain_scale, np.array(args.domain_min, dtype=np.float32))
    domain_bounds = np.array([model.aabb[0] - padding, model.aabb[1] + padding], dtype=np.float32)
    particles = ParticleSystem(
        ctx,
        particle_prog,
        count=args.particles,
        radius=args.size,
        bounds=domain_bounds,
        obstacle_bounds=model.aabb,
        base_wind=args.wind,
        drag_coeff=args.drag,
        turbulence=args.turbulence,
        buoyancy=args.buoyancy,
        gravity=args.gravity,
        boundary_restitution=args.boundary_restitution,
        obstacle_restitution=args.obstacle_restitution,
        obstacle_tangent_damping=args.obstacle_tangent_damping,
        kick_strength=args.kick_strength,
        config_path=args.particle_config,
    )

    camera = OrbitCamera(
        distance=max(6.0, float(np.linalg.norm(model.size) * 1.5)),
        target=Vector3(model.center)
    )
    scene_extent = max(float(np.max(model.size)), 2.0)
    guides = SceneGuides(ctx, line_prog, model.aabb, scene_extent)
    near_plane = max(0.01, scene_extent * 1e-4)
    far_plane = max(200.0, scene_extent * 20.0)

    paused = False
    imgui_pause = False
    last_time = glfw.get_time()
    kb_orbit_speed = 1.6
    kb_zoom_speed = 10.0
    runtime_max_dt = args.max_dt
    imgui_panel_visible = True

    imgui_api = None
    imgui_renderer = None
    if not args.no_imgui_panel:
        imgui_api, ImGuiGlfwRenderer, imgui_import_error = load_imgui_backend()
        if imgui_api is not None:
            try:
                imgui_api.create_context()
                imgui_renderer = ImGuiGlfwRenderer(window, attach_callbacks=False)
            except Exception as exc:  # noqa: BLE001
                print(f"In-window ImGui panel disabled: {exc}")
                imgui_renderer = None
        else:
            print(f"In-window ImGui panel unavailable: {imgui_import_error}")

    control_panel = None
    if imgui_renderer is None and not args.no_control_panel:
        control_panel = RuntimeControlPanel(args)

    def key_callback(window, key, scancode, action, mods):
        nonlocal paused, imgui_panel_visible
        if imgui_renderer is not None:
            imgui_renderer.keyboard_callback(window, key, scancode, action, mods)
        if action == glfw.PRESS:
            if imgui_renderer is not None and imgui_api.get_io().want_capture_keyboard and key != glfw.KEY_ESCAPE:
                return
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            if key == glfw.KEY_H and imgui_renderer is not None:
                imgui_panel_visible = not imgui_panel_visible
            if key == glfw.KEY_SPACE:
                paused = not paused
            if key == glfw.KEY_R:
                particles.reset(bounds=domain_bounds)
            if key in (glfw.KEY_F, glfw.KEY_ENTER):
                particles.kick(direction=camera.forward_vector())
    glfw.set_key_callback(window, key_callback)

    def char_callback(window, codepoint):
        if imgui_renderer is not None:
            imgui_renderer.char_callback(window, codepoint)
    glfw.set_char_callback(window, char_callback)

    def scroll_cb(win, x_off, y_off):
        if imgui_renderer is not None:
            imgui_renderer.scroll_callback(win, x_off, y_off)
        if imgui_renderer is not None and imgui_api.get_io().want_capture_mouse:
            return
        camera.scroll(y_off)
    glfw.set_scroll_callback(window, scroll_cb)

    def mouse_cb(win, x, y):
        if imgui_renderer is not None:
            imgui_renderer.mouse_callback(win, x, y)
        if imgui_renderer is not None and imgui_api.get_io().want_capture_mouse:
            return
        camera.drag(win)
    glfw.set_cursor_pos_callback(window, mouse_cb)

    def mouse_button_cb(win, button, action, mods):
        if imgui_renderer is not None:
            imgui_renderer.mouse_button_callback(win, button, action, mods)
        if imgui_renderer is not None and imgui_api.get_io().want_capture_mouse:
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            was_click = camera.handle_mouse(win, action, mods)
            if was_click and action == glfw.RELEASE:
                x, y = glfw.get_cursor_pos(win)
                fb_w, fb_h = glfw.get_framebuffer_size(win)
                if fb_w > 0 and fb_h > 0:
                    proj = Matrix44.perspective_projection(60.0, fb_w / float(fb_h), near_plane, far_plane)
                    view = camera.view_matrix()
                    ray_o, ray_d = _pick_ray_from_mouse(x, y, fb_w, fb_h, proj, view)
                    hit = _ray_aabb_hit(ray_o, ray_d, model.aabb)
                    if hit is not None:
                        particles.impulse_at_point(
                            hit,
                            direction=ray_d,
                            strength=particles.kick_strength * 1.15,
                            radius=max(0.9, scene_extent * 0.35),
                        )
        if button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
            particles.kick(direction=camera.forward_vector())
    glfw.set_mouse_button_callback(window, mouse_button_cb)

    while not glfw.window_should_close(window):
        ui_pause = imgui_pause
        frame_max_dt = runtime_max_dt

        if imgui_renderer is not None:
            imgui_renderer.process_inputs()
            imgui_api.new_frame()
            if imgui_panel_visible:
                imgui_api.set_next_window_pos((10.0, 10.0), imgui_api.Cond_.first_use_ever)
                imgui_api.set_next_window_size((355.0, 460.0), imgui_api.Cond_.first_use_ever)
                imgui_api.begin("Flow Controls")

                changed, wind_x = imgui_api.slider_float("Wind X", float(particles.base_wind[0]), -4.0, 4.0)
                if changed:
                    particles.base_wind[0] = wind_x
                changed, wind_y = imgui_api.slider_float("Wind Y", float(particles.base_wind[1]), -2.0, 2.0)
                if changed:
                    particles.base_wind[1] = wind_y
                changed, wind_z = imgui_api.slider_float("Wind Z", float(particles.base_wind[2]), -4.0, 4.0)
                if changed:
                    particles.base_wind[2] = wind_z

                changed, turbulence = imgui_api.slider_float("Turbulence", float(particles.turbulence), 0.0, 3.0)
                if changed:
                    particles.turbulence = turbulence
                changed, drag = imgui_api.slider_float("Drag", float(particles.drag_coeff), 0.0, 3.0)
                if changed:
                    particles.drag_coeff = drag
                changed, buoyancy = imgui_api.slider_float("Buoyancy", float(particles.buoyancy), 0.0, 2.0)
                if changed:
                    particles.buoyancy = buoyancy
                changed, gravity = imgui_api.slider_float("Gravity", float(particles.gravity), 0.0, 2.0)
                if changed:
                    particles.gravity = gravity

                mean_radius = float(np.mean(particles.radius)) if particles.count > 0 else args.size
                changed, particle_size = imgui_api.slider_float("Particle Size", mean_radius, 0.02, 0.20)
                if changed:
                    particles.set_uniform_radius(particle_size)
                changed, kick_strength = imgui_api.slider_float("Kick Strength", float(particles.kick_strength), 0.1, 8.0)
                if changed:
                    particles.kick_strength = kick_strength
                changed, runtime_max_dt = imgui_api.slider_float("Max dt", float(runtime_max_dt), 0.005, 0.05)
                if changed:
                    runtime_max_dt = max(0.001, runtime_max_dt)
                    frame_max_dt = runtime_max_dt

                changed, imgui_pause = imgui_api.checkbox("Pause", imgui_pause)
                if changed:
                    ui_pause = imgui_pause

                if imgui_api.button("Gust"):
                    particles.kick(direction=camera.forward_vector())
                imgui_api.same_line()
                if imgui_api.button("Reset Particles"):
                    particles.reset(bounds=domain_bounds)

                imgui_api.separator()
                imgui_api.text(f"Model: {model.label}")
                if model.is_fallback:
                    imgui_api.text("Mode: fallback cube")
                imgui_api.text("Yellow box: obstacle bounds")
                imgui_api.text("Axes: X red, Y green, Z blue")
                imgui_api.text("H: hide/show panel")
                imgui_api.text("LMB click on object: local impulse")
                imgui_api.end()

        if control_panel is not None:
            ui_values = control_panel.poll()
            if ui_values is not None:
                particles.base_wind = np.array(ui_values["wind"], dtype=np.float32)
                particles.turbulence = ui_values["turbulence"]
                particles.drag_coeff = ui_values["drag"]
                particles.buoyancy = ui_values["buoyancy"]
                particles.gravity = ui_values["gravity"]
                particles.set_uniform_radius(ui_values["size"])
                particles.kick_strength = ui_values["kick_strength"]
                frame_max_dt = ui_values["max_dt"]
                runtime_max_dt = frame_max_dt
                ui_pause = ui_values["pause"]
                if ui_values["reset"]:
                    particles.reset(bounds=domain_bounds)
                if ui_values["gust"]:
                    particles.kick(direction=camera.forward_vector())

        now = glfw.get_time()
        dt = min(now - last_time, frame_max_dt)
        last_time = now

        if not paused and not ui_pause:
            particles.step(dt)

        # Touchpad-friendly camera controls without mouse drag.
        if (
            glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_A) == glfw.PRESS
        ):
            camera.orbit(-kb_orbit_speed * dt, 0.0)
        if (
            glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_E) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_D) == glfw.PRESS
        ):
            camera.orbit(kb_orbit_speed * dt, 0.0)
        if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS or glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            camera.orbit(0.0, -kb_orbit_speed * dt)
        if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS or glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            camera.orbit(0.0, kb_orbit_speed * dt)
        if glfw.get_key(window, glfw.KEY_EQUAL) == glfw.PRESS or glfw.get_key(window, glfw.KEY_KP_ADD) == glfw.PRESS:
            camera.scroll(kb_zoom_speed * dt)
        if glfw.get_key(window, glfw.KEY_MINUS) == glfw.PRESS or glfw.get_key(window, glfw.KEY_KP_SUBTRACT) == glfw.PRESS:
            camera.scroll(-kb_zoom_speed * dt)

        w, h = glfw.get_framebuffer_size(window)
        if w == 0 or h == 0:
            glfw.poll_events()
            continue
        ctx.viewport = (0, 0, w, h)
        ctx.clear(*BACKGROUND, 1.0)

        proj = Matrix44.perspective_projection(60.0, w / float(h), near_plane, far_plane)
        view = camera.view_matrix()

        light_dir = Vector3([0.6, 1.0, 0.5]).normalized
        guides.draw(proj, view)
        model.draw(proj, view, light_dir)
        # Keep particles visible even when they overlap the obstacle.
        ctx.disable(moderngl.DEPTH_TEST)
        particles.draw(proj, view, light_dir)
        ctx.enable(moderngl.DEPTH_TEST)
        if imgui_renderer is not None:
            imgui_api.render()
            imgui_renderer.render(imgui_api.get_draw_data())

        glfw.swap_buffers(window)
        glfw.poll_events()

    if control_panel is not None and control_panel.active:
        control_panel.close()
    if imgui_renderer is not None:
        imgui_renderer.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
