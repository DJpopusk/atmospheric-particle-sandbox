import numpy as np


def _as_radius_array(radius, n: int):
    if np.isscalar(radius):
        return np.full((n,), float(radius), dtype=np.float32)
    arr = np.asarray(radius, dtype=np.float32)
    if len(arr) != n:
        raise ValueError("radius array size does not match particle count")
    return arr


def collide_domain_aabb(pos: np.ndarray, vel: np.ndarray, bounds, radius, restitution: float = 0.55):
    rad = _as_radius_array(radius, len(pos))
    for axis in range(3):
        lo = bounds[0][axis] + rad
        hi = bounds[1][axis] - rad
        low_mask = pos[:, axis] < lo
        high_mask = pos[:, axis] > hi
        if low_mask.any():
            pos[low_mask, axis] = lo[low_mask]
            vel[low_mask, axis] *= -restitution
        if high_mask.any():
            pos[high_mask, axis] = hi[high_mask]
            vel[high_mask, axis] *= -restitution


def collide_obstacle_aabb(
    pos: np.ndarray,
    vel: np.ndarray,
    obstacle_bounds,
    radius,
    restitution: float = 0.25,
    tangent_damping: float = 0.85,
):
    rad = _as_radius_array(radius, len(pos))
    lo = obstacle_bounds[0][None, :] - rad[:, None]
    hi = obstacle_bounds[1][None, :] + rad[:, None]
    inside = np.logical_and(pos >= lo, pos <= hi).all(axis=1)
    indices = np.where(inside)[0]
    for i in indices:
        px, py, pz = pos[i]
        li = lo[i]
        hi_i = hi[i]
        distances = np.array(
            [
                abs(px - li[0]),
                abs(hi_i[0] - px),
                abs(py - li[1]),
                abs(hi_i[1] - py),
                abs(pz - li[2]),
                abs(hi_i[2] - pz),
            ],
            dtype=np.float32,
        )
        face = int(np.argmin(distances))
        normal = np.zeros(3, dtype=np.float32)
        if face == 0:
            pos[i, 0] = li[0]
            normal[0] = -1.0
        elif face == 1:
            pos[i, 0] = hi_i[0]
            normal[0] = 1.0
        elif face == 2:
            pos[i, 1] = li[1]
            normal[1] = -1.0
        elif face == 3:
            pos[i, 1] = hi_i[1]
            normal[1] = 1.0
        elif face == 4:
            pos[i, 2] = li[2]
            normal[2] = -1.0
        else:
            pos[i, 2] = hi_i[2]
            normal[2] = 1.0

        v = vel[i]
        vn = float(np.dot(v, normal))
        vt = v - vn * normal
        if vn < 0.0:
            vn = -vn * restitution
        vel[i] = vt * tangent_damping + vn * normal


def apply_atmospheric_forces(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    dt: float,
    time_s: float,
    base_wind=(1.2, 0.08, 0.0),
    drag_coeff: float = 0.9,
    turbulence: float = 0.7,
    buoyancy: float = 0.25,
    gravity: float = 0.10,
):
    base = np.array(base_wind, dtype=np.float32)
    x = pos[:, 0] * 0.6 + time_s * 0.7
    y = pos[:, 1] * 0.8 - time_s * 0.4
    z = pos[:, 2] * 0.7 + time_s * 0.5
    swirl = np.stack(
        [
            np.sin(y) + 0.5 * np.cos(z),
            np.sin(z) - 0.5 * np.cos(x),
            np.sin(x) + 0.5 * np.cos(y),
        ],
        axis=1,
    ).astype(np.float32)
    flow = base + turbulence * swirl
    accel = (flow - vel) * (drag_coeff / mass[:, None])
    thermal = buoyancy * np.exp(-0.18 * pos[:, 1] * pos[:, 1])
    accel[:, 1] += thermal - gravity
    vel += accel * dt


def _hash_positions(pos: np.ndarray, cell_size: float):
    inv = 1.0 / max(cell_size, 1e-5)
    keys = np.floor(pos * inv).astype(np.int32)
    buckets = {}
    for i, key in enumerate(keys):
        k = (int(key[0]), int(key[1]), int(key[2]))
        buckets.setdefault(k, []).append(i)
    return keys, buckets


def _neighbor_keys(key):
    kx, ky, kz = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (kx + dx, ky + dy, kz + dz)


def collide_particles_hashed(
    pos: np.ndarray,
    vel: np.ndarray,
    radius,
    mass: np.ndarray,
    restitution: float = 0.75,
    cell_size: float | None = None,
):
    n = len(pos)
    if n < 2:
        return
    rad = _as_radius_array(radius, n)
    if cell_size is None:
        cell_size = float(np.max(rad) * 2.2)
    keys, buckets = _hash_positions(pos, cell_size)
    inv_mass = 1.0 / np.maximum(mass.astype(np.float32), 1e-6)

    for i in range(n):
        key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        for nkey in _neighbor_keys(key):
            neigh = buckets.get(nkey)
            if neigh is None:
                continue
            for j in neigh:
                if j <= i:
                    continue
                delta = pos[i] - pos[j]
                dist2 = float(delta.dot(delta))
                min_dist = float(rad[i] + rad[j])
                if dist2 >= min_dist * min_dist or dist2 < 1e-10:
                    continue

                dist = dist2 ** 0.5
                nrm = delta / dist
                overlap = min_dist - dist

                w_i = float(inv_mass[i])
                w_j = float(inv_mass[j])
                w_sum = w_i + w_j
                if w_sum <= 1e-10:
                    continue

                pos[i] += nrm * (overlap * (w_i / w_sum))
                pos[j] -= nrm * (overlap * (w_j / w_sum))

                rel = vel[i] - vel[j]
                sep = float(rel.dot(nrm))
                if sep > 0.0:
                    continue
                impulse = -(1.0 + restitution) * sep / w_sum
                vel[i] += nrm * (impulse * w_i)
                vel[j] -= nrm * (impulse * w_j)


def run_atmosphere_self_check(
    steps: int = 480,
    count: int = 600,
    dt: float = 1.0 / 60.0,
    seed: int = 7,
    base_wind=(1.2, 0.08, 0.0),
    drag_coeff: float = 0.9,
    turbulence: float = 0.7,
    buoyancy: float = 0.25,
    gravity: float = 0.10,
    boundary_restitution: float = 0.55,
    obstacle_restitution: float = 0.25,
    obstacle_tangent_damping: float = 0.85,
    radius: float = 0.08,
):
    rng = np.random.default_rng(seed)
    domain = np.array([[-5.0, -3.0, -5.0], [5.0, 3.0, 5.0]], dtype=np.float32)
    obstacle = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    span = domain[1] - domain[0]
    pos = domain[0] + rng.random((count, 3), dtype=np.float32) * span
    vel = (rng.random((count, 3), dtype=np.float32) - 0.5) * 0.6
    mass = 0.8 + 0.4 * rng.random(count, dtype=np.float32)
    rad = np.full((count,), radius, dtype=np.float32)

    for _ in range(8):
        inside = np.logical_and(pos >= obstacle[0], pos <= obstacle[1]).all(axis=1)
        if not inside.any():
            break
        pos[inside] = domain[0] + rng.random((inside.sum(), 3), dtype=np.float32) * span

    time_s = 0.0
    for _ in range(steps):
        apply_atmospheric_forces(
            pos,
            vel,
            mass,
            dt,
            time_s,
            base_wind=base_wind,
            drag_coeff=drag_coeff,
            turbulence=turbulence,
            buoyancy=buoyancy,
            gravity=gravity,
        )
        collide_particles_hashed(pos, vel, rad, mass, restitution=0.75)
        pos += vel * dt
        collide_domain_aabb(pos, vel, domain, rad, restitution=boundary_restitution)
        collide_obstacle_aabb(
            pos,
            vel,
            obstacle,
            rad,
            restitution=obstacle_restitution,
            tangent_damping=obstacle_tangent_damping,
        )
        if not np.isfinite(pos).all() or not np.isfinite(vel).all():
            return {"ok": False, "reason": "non-finite values during integration"}
        time_s += dt

    inside_after = np.logical_and(pos >= obstacle[0], pos <= obstacle[1]).all(axis=1).mean()
    mean_speed = np.linalg.norm(vel, axis=1).mean()
    return {
        "ok": True,
        "particles": count,
        "steps": steps,
        "mean_speed": float(mean_speed),
        "inside_obstacle_ratio": float(inside_after),
    }
