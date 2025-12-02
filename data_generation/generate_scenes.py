# habitat_hm3d_multicam_dynamic.py
import os, numpy as np, imageio, argparse, pathlib
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors, quat_from_magnum
import magnum as mn
import random

def score_center(
    sim: habitat_sim.Simulator,
    center_xyz: np.ndarray,
    num_cams: int = 10,
    cam_radius: float = 0.7,
    cam_height: float = 2.0,
    depth_uuid: str = "depth",
    max_valid_depth: float = 20.0,
) -> float:
    """
    Put `num_cams` virtual viewpoints on a ring around `center_xyz`,
    look at the center, render depth, and return how much stuff is visible.
    Higher = better.
    """
    scores = []
    for i in range(num_cams):
        theta = 2.0 * np.pi * i / num_cams
        cam_pos = np.array([
            center_xyz[0] + cam_radius * np.cos(theta),
            cam_height,
            center_xyz[2] + cam_radius * np.sin(theta),
        ], dtype=np.float32)

        st = habitat_sim.AgentState()
        st.position = cam_pos

        # orient toward center
        q_cam = look_at_quat(cam_pos, center_xyz)
        st.rotation = quat_from_magnum(q_cam)
        sim.get_agent(0).set_state(st)   # use agent 0 as a temporary renderer

        obs = sim.get_sensor_observations(0)
        if depth_uuid not in obs:
            continue
        depth = obs[depth_uuid]  # HxW float32 meters
        depth = np.asarray(depth, dtype=np.float32)

        # 1) simple coverage: how many pixels are finite and not too far
        valid = np.logical_and(np.isfinite(depth), depth < max_valid_depth)
        coverage = valid.mean()

        # 2) (optional) center-facing check: does the ray toward the center hit at about the right distance?
        # expected_dist = np.linalg.norm(cam_pos - center_xyz)
        # h, w = depth.shape
        # center_depth = depth[h//2, w//2]
        # facing = float(abs(center_depth - expected_dist) < 0.3 * expected_dist)

        scores.append(coverage)  # or 0.7*coverage + 0.3*facing
    if not scores:
        return 0.0
    return float(np.mean(scores))


def find_best_center(
    sim: habitat_sim.Simulator,
    n_samples: int = 80,
    y_override: float = 1.5,
    **score_kwargs,
) -> np.ndarray:
    """
    Sample n navigable points, score them, return the best one.
    """
    pf = sim.pathfinder
    best_score = -1.0
    best_center = None
    for _ in range(n_samples):
        p = pf.get_random_navigable_point()
        p = np.array(p, dtype=np.float32)
        if y_override is not None:
            p[1] = y_override   # keep cameras at a nice visualization height
        s = score_center(sim, p, **score_kwargs)
        if s > best_score:
            best_score = s
            best_center = p
    return best_center


def find_sorted_centers(
    sim: habitat_sim.Simulator,
    n_samples: int = 80,
    y_override: float = 1.5,
    **score_kwargs,
) -> list[tuple[float, np.ndarray]]:
    """
    Sample n navigable points, score them, and return a sorted list of (score, point),
    highest score first.
    """
    print("here1")
    pf = sim.pathfinder
    print("here2")
    scored_points = []
    print("here3")
    for m in range(n_samples):
        print(m)
        p = pf.get_random_navigable_point()
        print("point")
        p = np.array(p, dtype=np.float32)
        if y_override is not None:
            p[1] = y_override
        print("scoring")
        s = score_center(sim, p, **score_kwargs)
        scored_points.append((s, p))

    # Sort by score, descending
    scored_points.sort(key=lambda x: x[0], reverse=True)
    return scored_points



def make_rgb_spec(uuid, res=(720, 1280), hfov=90.0):
    s = habitat_sim.CameraSensorSpec()           # not SensorSpec
    s.uuid = uuid
    s.sensor_type = habitat_sim.SensorType.COLOR
    s.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    h, w = int(res[0]), int(res[1])              # Habitat expects [H, W]
    s.resolution = [h, w]
    s.hfov = float(hfov)
    s.position = mn.Vector3(0.0, 0.0, 0.0)
    s.orientation = mn.Vector3(0.0, 0.0, 0.0)
    return s

def make_depth_spec(uuid="depth", res=(720, 1280), hfov=90.0, min_depth=0.05, max_depth=20.0):
    s = habitat_sim.CameraSensorSpec()
    s.uuid = uuid
    s.sensor_type = habitat_sim.SensorType.DEPTH
    s.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    h, w = int(res[0]), int(res[1])          # Habitat wants [H, W]
    s.resolution = [h, w]
    s.hfov = float(hfov)                     # degrees
    s.position = mn.Vector3(0.0, 0.0, 0.0)   # relative to agent
    s.orientation = mn.Vector3(0.0, 0.0, 0.0)
    s.min_depth = float(min_depth)           # meters
    s.max_depth = float(max_depth)           # meters
    return s

    
    
def mn_quat_to_xyzw(q: mn.Quaternion) -> np.ndarray:
    # return np.array([float(q.vector.x), float(q.vector.y), float(q.vector.z), float(q.scalar)], dtype=np.float32)
    return np.array([float(q.scalar), float(q.vector.x), float(q.vector.y), float(q.vector.z)], dtype=np.float32)

# def look_at_quat(from_p, to_p) -> mn.Quaternion:
#     f = np.asarray(to_p, np.float32) - np.asarray(from_p, np.float32)
#     n = float(np.linalg.norm(f))
#     if n < 1e-8:
#         print("identity")
#         return mn.Quaternion()  # identity
#     f /= n
#     # Habitat forward = +Z, up = +Y
#     yaw   = np.arctan2(float(f[0]), float(f[2]))        # rotate around +Y
#     pitch = -np.arcsin(np.clip(float(f[1]), -1.0, 1.0)) # then around +X
#     q_yaw   = mn.Quaternion.rotation(mn.Rad(yaw),   mn.Vector3.y_axis())
#     q_pitch = mn.Quaternion.rotation(mn.Rad(pitch), mn.Vector3.x_axis())
#     return q_yaw * q_pitch


def look_at_quat(from_p, to_p) -> mn.Quaternion:
    f = np.asarray(to_p, np.float32) - np.asarray(from_p, np.float32)
    n = float(np.linalg.norm(f))
    if n < 1e-8:
        # Degenerate: keep current orientation (identity)
        return mn.Quaternion()
    f /= n

    # Habitat: +Z forward, +Y up (right-handed)
    # 1) Yaw from world-space forward
    yaw = -np.arctan2(float(f[0]), -float(f[2]))                  # rotate around +Y

    q_yaw = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())
    
    # 2) Remove yaw before computing pitch
    f_vec = mn.Vector3(float(f[0]), float(f[1]), float(f[2]))
    f_no_yaw = q_yaw.inverted().transform_vector(f_vec)

    # Guard against tiny z due to numeric noise
    if abs(float(f_no_yaw.z)) < 1e-8:
        f_no_yaw = mn.Vector3(f_no_yaw.x, f_no_yaw.y, 1e-8 if f_no_yaw.z >= 0 else -1e-8)

    # 3) Pitch in yaw-neutral frame (keep roll = 0)
    # Looking up should be negative rotation around +X in Habitat
    # pitch = -np.arctan2(float(f_no_yaw.y), float(f_no_yaw.z))   # rotate around +X
    # pitch = np.arctan2(f_no_yaw.y, np.sqrt(f_no_yaw.x**2 + f_no_yaw.z**2))
    pitch = np.arctan2(float(f[1]), np.sqrt(float(f[0])**2 + float(f[2])**2))


    q_pitch = mn.Quaternion.rotation(mn.Rad(pitch), mn.Vector3.x_axis())
   

    # Apply pitch first, then yaw (rightmost first)
    return q_yaw * q_pitch

import json
import numpy as np

def intrinsics_from_hfov(h, w, hfov_deg):
    hfov = np.deg2rad(hfov_deg)
    fx = (w / 2.0) / np.tan(hfov * 0.5)
    vfov = 2.0 * np.arctan((h / w) * np.tan(hfov * 0.5))
    fy = (h / 2.0) / np.tan(vfov * 0.5)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    return fx, fy, cx, cy
import numpy as np

def quat_wxyz_to_R(qw, qx, qy, qz):
    """Return a 3x3 rotation matrix from a (w,x,y,z) quaternion."""
    # normalized?
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0.0:  # identity
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [    2*(xy + wz),  1 - 2*(xx + zz),        2*(yz - wx)],
        [    2*(xz - wy),      2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float32)

def backproject_depth_to_cam(D, fx, fy, cx, cy):
    """D: HxW (meters) -> (N,3) camera-frame points with Z>0."""
    H, W = D.shape
    ys, xs = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    Z = D.reshape(-1)
    X = (xs.reshape(-1) - cx) * Z / fx
    Y = (ys.reshape(-1) - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)  # (N,3)


def cam_to_world_points(P_cam, c2w_4x4):
    """(N,3) in camera -> (N,3) in world using 4x4 camera_to_world"""
    N = P_cam.shape[0]
    homog = np.concatenate([P_cam, np.ones((N,1), dtype=P_cam.dtype)], axis=1)
    Pw = (homog @ c2w_4x4.T)[:, :3]
    return Pw

def write_ply_rgb(path, points, colors=None):
    """points: (N,3) float; colors: (N,3) uint8 or None"""
    pts = points[np.isfinite(points).all(1)]
    if colors is not None:
        colors = colors[:pts.shape[0]]
    with open(path, "w") as f:
        if colors is None:
            f.write(
                "ply\nformat ascii 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n"
            )
            for x,y,z in pts:
                f.write(f"{x} {y} {z}\n")
        else:
            f.write(
                "ply\nformat ascii 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            )
            for (x,y,z), (r,g,b) in zip(pts, colors):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")






def project_point_to_pixel(Pw, R_c2w, t_c2w, fx, fy, cx, cy, W, H):
    # camera_to_world -> world_to_camera
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    Pc = R_w2c @ Pw + t_w2c
    z = float(Pc[2])
    if z <= 0: 
        return None  # behind camera
    u = fx * (Pc[0] / z) + cx
    v = fy * (Pc[1] / z) + cy
    if u < 0 or v < 0 or u >= W or v >= H:
        return None
    return int(round(u)), int(round(v)), z


def place_object_visible_and_safe(
    sim,
    handle: str,
    num_cams: int,
    footprint_radius: float,
    half_height: float,
    fx, fy, cx, cy, w, h,
    depth_uuid: str = "depth",
    min_visible_cams: int = 2,     # <- require at least k cameras
    max_trials: int = 300,
    margin: float = 0.10,
    avoid_center:np.ndarray = None,
    avoid_radius: float = 0.0,
):
    pf = sim.pathfinder
    
    
    obj_mgr = sim.get_rigid_object_manager()
    obj = obj_mgr.add_object_by_template_handle(handle)
    if obj is None:
        return None
    
    for _ in range(max_trials):
        p = np.array(pf.get_random_navigable_point(), dtype=np.float32)

        # clearance from walls/obstacles
        d = pf.distance_to_closest_obstacle(p)
        if not np.isfinite(d) or d < (footprint_radius + margin):
            continue

        # optionally avoid camera ring center
        if avoid_center is not None:
            if np.linalg.norm(p[[0,2]] - np.array(avoid_center)[[0,2]]) < (avoid_radius + footprint_radius + margin):
                continue

        # place base on floor
        pos = mn.Vector3(float(p[0]), float(p[1] + half_height), float(p[2]))
        yaw = np.random.uniform(0, 2*np.pi)
        q_yaw = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())

        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.rotation = q_yaw
        obj.translation = pos

        # visibility test at the object center (you can offset to top if tall)
        Pw = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        seen = count_visible_cameras(
            sim, num_cams, Pw, depth_uuid, fx, fy, cx, cy, w, h, min_needed=min_visible_cams
        )
        if seen >= min_visible_cams:
            break

    # obj_mgr.remove_object_by_id(obj.object_id)
    return obj

R_FIX = np.diag([1, -1, -1]).astype(np.float32)  # same as in your writer

def cam_extrinsics_from_agent_state(astate):
    # Returns camera_to_world (with your axis fix applied)
    pos = np.array(astate.position, dtype=np.float32)
    q = astate.rotation
    try:
        qw, qx, qy, qz = float(q.w), float(q.x), float(q.y), float(q.z)
    except AttributeError:
        qw, qx, qy, qz = map(float, q)
    R = quat_wxyz_to_R(qw, qx, qy, qz).astype(np.float32)
    R = R @ R_FIX
    return R, pos

def visible_in_camera(sim, cam_idx, Pw, depth_uuid, fx, fy, cx, cy, w, h, 
                      win=3, max_rel_err=0.15, max_abs_err=0.10):
    obs = sim.get_sensor_observations(cam_idx)
    if depth_uuid not in obs: 
        return False
    D = np.asarray(obs[depth_uuid], dtype=np.float32)  # HxW in meters

    astate = sim.get_agent(cam_idx).get_state()
    R, t = cam_extrinsics_from_agent_state(astate)

    proj = project_point_to_pixel(Pw, R, t, fx, fy, cx, cy, W=w, H=h)
    if proj is None: 
        return False
    u, v, zexp = proj

    # small window vote to be robust
    u0, u1 = max(0, u - win), min(w - 1, u + win)
    v0, v1 = max(0, v - win), min(h - 1, v + win)
    patch = D[v0:v1+1, u0:u1+1]
    if patch.size == 0 or not np.isfinite(patch).any():
        return False
    zobs = np.nanmin(patch)  # nearest thing in that patch

    # accept if observed depth matches expected distance (occlusion-aware)
    rel_ok = abs(zobs - zexp) <= max_rel_err * max(zexp, 1e-6)
    abs_ok = abs(zobs - zexp) <= max_abs_err
    return bool(rel_ok or abs_ok)

def count_visible_cameras(sim, num_cams, Pw, depth_uuid, fx, fy, cx, cy, w, h, min_needed=1):
    seen = 0
    for cam_idx in range(num_cams):
        if visible_in_camera(sim, cam_idx, Pw, depth_uuid, fx, fy, cx, cy, w, h):
            seen += 1
            if seen >= min_needed:
                break
    return seen

def _snap_to_navmesh(pf, p):
    # Try Habitat's snap if available; else do a tiny radial search
    if hasattr(pf, "snap_point"):
        sp = pf.snap_point(p)        # snaps to nearest navigable position
        return np.array(sp, dtype=np.float32)
    # Fallback: spiral search for a navigable point
    for r in np.linspace(0.0, 0.5, 8):
        for ang in np.linspace(0, 2*np.pi, 12, endpoint=False):
            q = np.array([p[0]+r*np.cos(ang), p[1], p[2]+r*np.sin(ang)], np.float32)
            if pf.is_navigable(q):
                return q
    return p  # last resort (may be non-navigable)

def _has_clearance(pf, p, footprint_radius, margin):
    d = pf.distance_to_closest_obstacle(p)
    return (np.isfinite(d) and d >= (footprint_radius + margin))


def place_object_near_center_visible(
    sim,
    handle: str,
    num_cameras:int,
    centre_cams_xyz,              # your auto-picked center [x,y,z]
    fx, fy, cx, cy, w, h,         # intrinsics + res you already compute
    depth_uuid="depth",
    min_visible_cams=2,           # require k cameras to see it
    footprint_radius=0.3,         # object footprint (XZ) in meters
    half_height=0.5,              # half height so it sits on the floor
    annulus_inner=0.6,            # search ring: inner radius from center
    annulus_outer=2.5,            # ...and outer radius
    max_trials=120,
    margin=0.05,                  # small buffer vs walls/obstacles
):
    pf = sim.pathfinder
    obj_mgr = sim.get_rigid_object_manager()
    obj = obj_mgr.add_object_by_template_handle(handle)
    if obj is None:
        return None

    center = np.array(centre_cams_xyz, dtype=np.float32)
    cand = None
    # quasi–blue-noise: random angle + mildly jittered radius
    for _ in range(max_trials):
        ang = np.random.uniform(0, 2*np.pi)
        r   = np.random.uniform(annulus_inner, annulus_outer)
        # small gaussian jitter to avoid regular patterns
        r  += np.random.normal(scale=0.05*(annulus_outer-annulus_inner))

        cand = np.array([center[0] + r*np.cos(ang), center[1], center[2] + r*np.sin(ang)], np.float32)
        cand = _snap_to_navmesh(pf, cand)

        if not pf.is_navigable(cand): 
            continue
        
        if not _has_clearance(pf, cand, footprint_radius, margin):
            continue

        # place (KINEMATIC) at floor + half height; random yaw
        pos = mn.Vector3(float(cand[0]), float(cand[1] + half_height), float(cand[2]))
        yaw = np.random.uniform(0, 2*np.pi)
        q_yaw = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3.y_axis())

        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.rotation = q_yaw
        obj.translation = pos

        # visibility test at object center (you can also test top: pos.y + half_height)
        Pw = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
        seen = count_visible_cameras(sim, num_cameras, Pw, depth_uuid, fx, fy, cx, cy, w, h,
                                     min_needed=min_visible_cams)
        if seen >= min_visible_cams:
            break
        
    print("cand", cand)

    return obj, cand


import math, uuid


import math

def _spawn_one_with_temp_scale(sim, base_handle: str, scale_min=0.5, scale_max=2.5):
    """
    Temporarily modify the template's scale, spawn ONE object, then restore the template.
    Returns (obj, approx_radius_xz, approx_half_height, scale_used).
    """
    tmpl_mgr = sim.get_object_template_manager()
    obj_mgr  = sim.get_rigid_object_manager()

    attr = tmpl_mgr.get_template_by_handle(base_handle)
    if attr is None:
        return None, None, None, None

    # Save original scale (Magnum Vector3–like)
    try:
        orig_scale = mn.Vector3(attr.scale)  # make a copy
    except Exception:
        # Fallback: treat as tuple/list
        sx, sy, sz = attr.scale
        orig_scale = mn.Vector3(float(sx), float(sy), float(sz))

    s = float(np.random.uniform(scale_min, scale_max))

    # Apply temporary uniform scale
    attr.scale = orig_scale * s
    tmpl_mgr.register_template(attr, base_handle)

    # Spawn ONE object with this scaled template
    obj = obj_mgr.add_object_by_template_handle(base_handle)

    # Restore template scale so later spawns can choose a different s
    attr.scale = orig_scale
    tmpl_mgr.register_template(attr, base_handle)

    # Rough size estimate (match your earlier 0.1 heuristic)
    base_half_extent = 0.1
    approx_half_height = base_half_extent * s
    approx_radius_xz   = math.sqrt(2.0) * base_half_extent * s

    return obj, approx_radius_xz, approx_half_height, s

def _clone_and_scale_template(tmpl_mgr, base_handle: str, scale_min=0.5, scale_max=2.5):
    """
    Clone an object template so each object can have its own scale
    without mutating the shared base template.
    Returns (new_handle, approx_radius_xz, approx_half_height, scale_factor).
    """
    base = tmpl_mgr.get_template_by_handle(base_handle)
    # Defensive copy: Habitat templates implement copy()
    t = base.clone()
    s = float(np.random.uniform(scale_min, scale_max))
    t.scale *= s

    new_handle = f"{base_handle}__rand_{uuid.uuid4().hex[:8]}"
    tmpl_mgr.register_template(t, new_handle)

    # Very rough footprint/height guess (your original code used ~0.1)
    # If your assets are unit primitives, these scale lines keep it consistent.
    # Adjust the 0.1 coefficients if your source units differ.
    base_half_extent = 0.1
    approx_half_height = base_half_extent * s
    approx_radius_xz   = math.sqrt(2.0) * base_half_extent * s
    return new_handle, approx_radius_xz, approx_half_height, s

def _random_traj(start_xyz):
    """
    Create a randomized trajectory function f(t, dt) -> (x,y,z).
    't' is integer step index; 'dt' is seconds per step.
    """
    kind = random.choice(["circle_xz", "pingpong_axis", "helix", "lissajous_xz"])

    if kind == "circle_xz":
        r = np.random.uniform(0.4, 1.2)
        omega = np.random.uniform(0.4, 1.6)  # rad/s
        phase = np.random.uniform(0, 2*np.pi)
        y0 = start_xyz[1]
        def f(t, dt):
            tt = t * dt
            x = start_xyz[0] + r * math.cos(omega*tt + phase)
            z = start_xyz[2] + r * math.sin(omega*tt + phase)
            return (x, y0, z)

    elif kind == "pingpong_axis":
        axis = random.choice([0, 1, 2])  # x / y / z
        span = np.random.uniform(1.5, 4.0)
        period = np.random.uniform(2.5, 6.0)  # seconds
        def f(t, dt):
            # triangle wave in [ -span/2, +span/2 ]
            tt = (t * dt) % period
            alpha = tt / (period / 2.0)
            alpha = 2.0 - alpha if tt > period/2.0 else alpha
            offset = (alpha - 0.5) * span
            p = list(start_xyz)
            p[axis] = start_xyz[axis] + offset
            return tuple(p)

    elif kind == "helix":
        r = np.random.uniform(0.4, 0.9)
        omega = np.random.uniform(0.8, 1.5)
        v_y = np.random.uniform(-0.15, 0.15)  # m/s up/down
        def f(t, dt):
            tt = t * dt
            x = start_xyz[0] + r * math.cos(omega*tt)
            z = start_xyz[2] + r * math.sin(omega*tt)
            y = start_xyz[1] + v_y * tt
            return (x, y, z)

    else:  # "lissajous_xz"
        ax = np.random.uniform(0.5, 1.5)
        az = np.random.uniform(0.5, 1.5)
        wx = np.random.uniform(0.6, 1.4)
        wz = np.random.uniform(0.6, 1.4)
        phx = np.random.uniform(0, 2*np.pi)
        phz = np.random.uniform(0, 2*np.pi)
        y0 = start_xyz[1]
        def f(t, dt):
            tt = t * dt
            x = start_xyz[0] + ax * math.sin(wx*tt + phx)
            z = start_xyz[2] + az * math.sin(wz*tt + phz)
            return (x, y0, z)

    return f

# --- add near your other imports ---
import traceback
from pathlib import Path
import random as _random
import hashlib


# ----------------- NEW: wrapper that runs your whole pipeline for one scene -----------------
def run_for_scene(scene_path: str, args, root_for_rel = None):

    out_root = args.out
    scene_path = os.path.abspath(os.path.expanduser(scene_path))
    
    scene_path = Path(scene_path).expanduser().resolve()

    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    # pick out subdir based on out_structure
    scene_stem = Path(scene_path).stem
    if getattr(args, "out_structure", "tree") == "tree" and root_for_rel:
        # rel = Path(scene_path).relative_to(root_for_rel)
        root_for_rel = Path(root_for_rel).resolve()
        rel = scene_path.relative_to(root_for_rel)
        out_sub = rel.with_suffix("")  # strip .glb
        out_dir = Path(out_root) / out_sub
    elif getattr(args, "out_structure", "tree") == "hashed" and root_for_rel:
        rel_str = str(Path(scene_path).relative_to(root_for_rel)).encode("utf-8")
        short = hashlib.sha1(rel_str).hexdigest()[:8]
        out_dir = Path(out_root) / f"{scene_stem}__{short}"
    else:  # flat
        out_dir = Path(out_root) / scene_stem

    out_dir.mkdir(parents=True, exist_ok=True)
    out = str(out_dir)
    Path(out).mkdir(parents=True, exist_ok=True)

    print("pick subdir")

    # --- Simulator config (unchanged) ---
    sim_cfg = habitat_sim.SimulatorConfiguration()
    # sim_cfg.scene_id = scene_path
    
    
    sim_cfg.scene_id = str(scene_path)   # <-- cast to string

    sim_cfg.gpu_device_id = -1
    sim_cfg.enable_physics = True


    cfg = habitat_sim.Configuration(sim_cfg, [])
    print("config habitat")
    for _ in range(args.num_cams):
        spec = make_rgb_spec("rgb", res=(args.height, args.width))
        spec_d = make_depth_spec("depth", res=(args.height, args.width), hfov=spec.hfov)
        cfg.agents.append(habitat_sim.AgentConfiguration(sensor_specifications=[spec, spec_d]))

    print("create cameras")
    sim = habitat_sim.Simulator(cfg)

    try:
        # >>>>>>>>>>>> everything from here down is exactly your existing logic <<<<<<<<<<<<
        # find_sorted_centers(...) ... time loop ... writing frames/poses/index
        sorted_centers = find_sorted_centers(sim,
            n_samples=100,
            y_override=1.5,
            num_cams=args.num_cams,
            cam_radius=args.cam_radius,
            cam_height=args.cam_height)
        print("sorted centers")
        MIN_SEEN = 3

        for i, (_, centre_cams) in enumerate(sorted_centers):
            print(f"[{scene_stem}] ============ center {i} ============")
            out_loop = os.path.join(out, f"{i}/")
            Path(out_loop).mkdir(parents=True, exist_ok=True)

            # ---- intrinsics once ----
            h, w = args.height, args.width
            hfov_deg = 90.0
            fx, fy, cx, cy = intrinsics_from_hfov(h, w, hfov_deg)

            with open(os.path.join(out_loop, "intrinsics.json"), "w") as f:
                json.dump({
                    "camera_model": "OPENCV",
                    "w": w, "h": h, "fl_x": fx, "fl_y": fy,
                    "cx": cx, "cy": cy, "hfov": hfov_deg
                }, f, indent=2)

            time_index = []

            # --- Object initial position (your args) ---
            obj0 = np.array([args.obj_x, args.obj_y, args.obj_z], dtype=np.float32)

            # place ring cameras looking at the center
            R = float(args.cam_radius)
            H = float(args.cam_height)
            NUM_CAMS = np.random.randint(4,16)
            for i_cam in range(NUM_CAMS):
                theta = 2.0 * np.pi * i_cam / NUM_CAMS
                pos = np.array([
                    centre_cams[0] + R * np.cos(theta),
                    H,
                    centre_cams[2] + R * np.sin(theta)
                ], dtype=np.float32)
                q_cam = look_at_quat(pos, centre_cams)
                st = habitat_sim.AgentState()
                st.position = [float(pos[0]), float(pos[1]), float(pos[2])]
                st.rotation = quat_from_magnum(q_cam)
                sim.get_agent(i_cam).set_state(st)

            # ---------- randomized objects ----------
            tmpl_mgr = sim.get_object_template_manager()
            obj_mgr  = sim.get_rigid_object_manager()

            NUM_OBJS = np.random.randint(1, 6)
            base_candidates = []
            for key in ["cylinder", "cube", "sphere", "box", "capsule"]:
                base_candidates += [h for h in tmpl_mgr.get_template_handles()
                                    if key in h.lower() and "solid" in h.lower()]
            if not base_candidates:
                base_candidates = tmpl_mgr.get_template_handles()

            dyn_objs = []
            inner = max(0.2, args.cam_radius * 0.6)
            outer = args.cam_radius * 3.0

            for _k in range(NUM_OBJS):
                base_handle = random.choice(base_candidates)
                placed_obj, radius_xz, half_h, scale_used = _spawn_one_with_temp_scale(
                    sim, base_handle, scale_min=1.5, scale_max=3
                )
                if placed_obj is None:
                    continue

                probe_obj, probe_pos = place_object_near_center_visible(
                    sim, base_handle, args.num_cams, centre_cams,
                    fx, fy, cx, cy, args.width, args.height,
                    depth_uuid="depth", min_visible_cams=MIN_SEEN,
                    footprint_radius=radius_xz, half_height=half_h,
                    annulus_inner=inner, annulus_outer=outer, max_trials=180
                )
                if probe_pos is None:
                    sim.get_rigid_object_manager().remove_object_by_id(placed_obj.object_id)
                    continue

                sim.get_rigid_object_manager().remove_object_by_id(probe_obj.object_id)
                placed_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                placed_obj.translation = mn.Vector3(float(probe_pos[0]),
                                                    float(probe_pos[1]),
                                                    float(probe_pos[2]))
                traj_fn = _random_traj((float(probe_pos[0]),
                                        float(probe_pos[1]),
                                        float(probe_pos[2])))
                dyn_objs.append({"obj": placed_obj, "traj": traj_fn})

            if not dyn_objs:
                print(f"[{scene_stem}] no objects placed for this center; skipping.")
                continue

            # ---------- TIME LOOP ----------
            T  = int(args.fps * args.secs)
            dt = 1.0 / float(args.fps)

            for t in range(T):
                time_dir = os.path.join(out_loop, f"time_{t:05d}")
                os.makedirs(time_dir, exist_ok=True)

                poses_this_time = []
                timestamp = float(t) / float(args.fps)

                for i_cam in range(args.num_cams):
                    obs = sim.get_sensor_observations(i_cam)

                    rgb = obs["rgb"][..., :3]
                    rgb_name = f"cam{i_cam}.jpg"
                    imageio.imwrite(os.path.join(time_dir, rgb_name), rgb)

                    depth_name = None
                    if "depth" in obs:
                        D = obs["depth"].astype(np.float32)
                        depth_name = f"cam{i_cam}_depth.npy"
                        np.save(os.path.join(time_dir, depth_name), D)

                    astate = sim.get_agent(i_cam).get_state()
                    pos = np.array(astate.position, dtype=np.float64)
                    q   = astate.rotation
                    try:
                        qw, qx, qy, qz = float(q.w), float(q.x), float(q.y), float(q.z)
                    except AttributeError:
                        qw, qx, qy, qz = map(float, q)
                    Rm = quat_wxyz_to_R(qw, qx, qy, qz).astype(np.float64)
                    R_fix = np.diag([1, -1, -1])
                    Rm = Rm @ R_fix
                    c2w = np.eye(4, dtype=np.float64)
                    c2w[:3, :3] = Rm
                    c2w[:3,  3] = pos

                    poses_this_time.append({
                        "camera_index": i_cam,
                        "file_path": rgb_name,
                        "depth_file": depth_name,
                        "transform_matrix": c2w.tolist(),
                    })

                with open(os.path.join(time_dir, "poses.json"), "w") as f:
                    json.dump({"timestamp": timestamp, "poses": poses_this_time}, f, indent=2)

                # move objects + step physics
                for entry in dyn_objs:
                    x, y, z = entry["traj"](t, dt)
                    entry["obj"].translation = mn.Vector3(float(x), float(y), float(z))
                sim.step_physics(dt)

            with open(os.path.join(out_loop, "index.json"), "w") as f:
                json.dump({"times": time_index}, f, indent=2)

        # >>>>>>>>>>>> end of your original logic <<<<<<<<<<<<
    finally:
        # make sure GPU/physics resources are released before the next scene
        try:
            sim.close()
        except Exception:
            pass
        del sim


# ----------------- MODIFY your argparse + main() -----------------
def main():
    ap = argparse.ArgumentParser()
    # single-scene (old) path still supported:
    ap.add_argument("--scene", help="Path to a single HM3D .glb (or dataset scene)")

    # NEW: batch processing
    ap.add_argument("--root", help="Root directory to search for scenes (recurses)")
    ap.add_argument("--pattern", default="*.glb", help="Glob to match scene files under --root")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N scenes (0 = all)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle scene order before processing")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip a scene if its output folder already exists and is non-empty")

    # rest of your original options:
    ap.add_argument("--out", default="out_frames")
    ap.add_argument("--num_cams", type=int, default=10)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--secs", type=int, default=15)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--obj_x", type=float, default=-5.0)
    ap.add_argument("--obj_y", type=float, default=0.2)
    ap.add_argument("--obj_z", type=float, default=-0.5)
    ap.add_argument("--cam_radius", type=float, default=0.7)
    ap.add_argument("--cam_height", type=float, default=2)
    ap.add_argument("--out_structure", choices=["tree", "flat", "hashed"], default="tree",
                help="How to lay out per-scene outputs. 'tree' mirrors folders under --root.")

    args = ap.parse_args()

    # choose mode
    if args.root:
        root = Path(os.path.expanduser(args.root))
        if not root.exists():
            raise FileNotFoundError(f"Root not found: {root}")

        scene_paths = sorted(p.as_posix() for p in root.rglob(args.pattern))
        if args.shuffle:
            _random.shuffle(scene_paths)
        if args.limit and args.limit > 0:
            scene_paths = scene_paths[:args.limit]

        if not scene_paths:
            raise FileNotFoundError(f"No scenes matched {args.pattern} under {root}")

        print(f"Found {len(scene_paths)} scene(s).")
        Path(args.out).mkdir(parents=True, exist_ok=True)

        # for idx, sp in enumerate(scene_paths, 1):
        #     scene_stem = Path(sp).stem
        #     out_dir = Path(args.out) / scene_stem
        #     if args.skip_existing and out_dir.exists():
        #         # consider “non-empty” as containing any file
        #         if any(out_dir.iterdir()):
        #             print(f"[{idx}/{len(scene_paths)}] SKIP existing: {scene_stem}")
        #             continue

        #     print(f"[{idx}/{len(scene_paths)}] Running scene: {sp}")
        #     try:
        #         run_for_scene(sp, args)
        #     except Exception as e:
        #         print(f"ERROR in scene {sp}: {e}")
        #         traceback.print_exc()
        #         # keep going to next scene
        #         continue
            
            
            


        print("Starting loop", flush=True)
        # --- loop (in main, batch mode) ---
        for idx, sp in enumerate(scene_paths, 1):
            # compute where we'd write, to honor --skip_existing consistently
            if args.out_structure == "tree":
                rel = Path(sp).relative_to(root).with_suffix("")
                out_dir = Path(args.out) / rel
            elif args.out_structure == "hashed":
                rel_str = str(Path(sp).relative_to(root)).encode("utf-8")
                short = hashlib.sha1(rel_str).hexdigest()[:8]
                out_dir = Path(args.out) / f"{Path(sp).stem}__{short}"
            else:
                out_dir = Path(args.out) / Path(sp).stem

            if args.skip_existing and out_dir.exists() and any(out_dir.iterdir()):
                print(f"[{idx}/{len(scene_paths)}] SKIP existing: {sp}")
                continue

            print(f"[{idx}/{len(scene_paths)}] Running scene: {sp}")
            try:
                run_for_scene(sp, args, root_for_rel=str(root))
            except Exception as e:
                print(f"ERROR in scene {sp}: {e}")
                traceback.print_exc()
                continue

        print("All done.")
        return

    # fallback: single-scene mode (original behavior)
    if not args.scene:
        raise SystemExit("Please pass either --scene <file> or --root <dir>.")
    run_for_scene(args.scene, args)

if __name__ == "__main__":
    main()
