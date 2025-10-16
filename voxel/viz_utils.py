from __future__ import annotations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt


def visualize_vggt_pointcloud(
    predictions,
    *,
    key="world_points",
    conf_key="world_points_conf",
    threshold=50.0,
    max_points=500_000,
    z_band=None,
    frame_stride=1,
):
    """
    Colorized VGGT point cloud with coordinate axes.
    Accepts images in (S,H,W,3) or (S,3,H,W).
    """
    import numpy as np
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("Open3D is required (pip install open3d).") from e

    def to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    if key not in predictions or "images" not in predictions:
        raise KeyError("predictions must contain key and 'images'")

    # World points: (S,H,W,3)
    WPTS = to_numpy(predictions[key])
    if WPTS.ndim != 4 or WPTS.shape[-1] != 3:
        raise ValueError(f"key expected (S,H,W,3), got {WPTS.shape}")
    S, H, W = WPTS.shape[:3]

    # Images: accept (S,H,W,3) OR (S,3,H,W)
    IMGS = to_numpy(predictions["images"])
    if IMGS.ndim == 4 and IMGS.shape[-1] == 3:
        pass  # already (S,H,W,3)
    elif IMGS.ndim == 4 and IMGS.shape[1] == 3:
        IMGS = np.transpose(IMGS, (0, 2, 3, 1))  # (S,3,H,W) -> (S,H,W,3)
    else:
        raise ValueError(f"images must be (S,H,W,3) or (S,3,H,W), got {IMGS.shape}")

    # Ensure dims match world points
    if IMGS.shape[:3] != (S, H, W):
        raise ValueError(f"images shape {IMGS.shape} must match key {(S,H,W,3)}")

    # Flatten selected frames
    frames = np.arange(0, S, max(1, int(frame_stride)))
    P = WPTS[frames].reshape(-1, 3).astype(np.float32)
    C = IMGS[frames].reshape(-1, 3).astype(np.float32)

    # Normalize colors to [0,1]
    if C.max() > 1.0:  # likely uint8
        C = C / 255.0
    C = np.clip(C, 0.0, 1.0)
    
    conf = np.array(predictions[conf_key][frames])
    conf = conf.reshape(-1).astype(np.float32)
    
    if threshold == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, threshold)


    valid = (~np.isnan(P).any(axis=1)) & np.isfinite(P).all(axis=1)
    valid = valid & (conf >= conf_threshold) & (conf > 1e-5)

    P = P[valid]; C = C[valid]

    # Optional z-band filter
    if z_band is not None:
        z0, z1 = float(z_band[0]), float(z_band[1])
        in_band = (P[:, 2] >= z0) & (P[:, 2] <= z1)
        P = P[in_band]; C = C[in_band]

    # Downsample
    n = P.shape[0]
    if n == 0:
        raise RuntimeError("No valid points to visualize (after filtering).")
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        P = P[idx]; C = C[idx]

    print(f"Visualizing {P.shape[0]} points from {len(frames)} frame(s)...")

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64))

    # Add coordinate frame (X=red, Y=green, Z=blue)
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,  # scale (meters)
        origin=[0, 0, 0]
    )

    # Visualize together
    o3d.visualization.draw_geometries(
        [pcd, axis_frame],
        window_name="VGGT Colored Point Cloud with Axes"
    )
    
def visualize_bev(bev: np.ndarray, meta: dict, title: str = "BEV occupancy (0.10 m)"):
    """Quick Matplotlib visualization of the BEV occupancy grid.
    -1 = unknown (dark), 0 = free, 100 = occupied
    """
    H, W = bev.shape
    res = float(meta.get("resolution", 0.10))
    ox, oy = meta.get("origin_xy", (0.0, 0.0))
    extent = [ox, ox + W * res, oy, oy + H * res]
    plt.figure()
    plt.imshow(bev, origin="lower", extent=extent)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.colorbar(label="occupancy value")
    plt.tight_layout()
    plt.show()

def visualize_voxels(vox,
                     z_band=(0.05, 1.8),
                     max_dim=96,          # clamp to keep it light
                     title="Occupied voxels (cubes)"):

    # Ensure occupied voxels list is NumPy-compatible
    occ_ijks = vox.occupied_voxels(zmin=float(z_band[0]), zmax=float(z_band[1]))
    occ_ijks = [tuple(int(x) for x in ijk) for ijk in occ_ijks]  # make sure they are tuples of ints

    if not occ_ijks:
        print("No occupied voxels to show.")
        return

    # Find tight bounds in index space
    I = np.array(occ_ijks, dtype=np.int32)
    imin, jmin, kmin = I.min(axis=0)
    imax, jmax, kmax = I.max(axis=0)

    # Make sure origin and voxel size are NumPy scalars
    origin = np.asarray(vox.origin, dtype=np.float32)
    vs = float(vox.p.voxel_size)

    # Clamp the block size (avoid huge dense arrays)
    ni, nj, nk = (imax - imin + 1), (jmax - jmin + 1), (kmax - kmin + 1)
    if max(ni, nj, nk) > max_dim:
        # Center a cropped window
        ci, cj, ck = (imin + imax) // 2, (jmin + jmax) // 2, (kmin + kmax) // 2
        half = max_dim // 2
        imin, imax = ci - half, ci + half
        jmin, jmax = cj - half, cj + half
        kmin, kmax = ck - half, ck + half
        ni, nj, nk = (imax - imin + 1), (jmax - jmin + 1), (kmax - kmin + 1)

    # Fill occupancy grid
    filled = np.zeros((ni, nj, nk), dtype=bool)
    for (i, j, k) in occ_ijks:
        if imin <= i <= imax and jmin <= j <= jmax and kmin <= k <= kmax:
            filled[i - imin, j - jmin, k - kmin] = True

    # Compute voxel corner coordinates in meters
    xe = np.arange(imin, imax + 2) * vs + origin[0]
    ye = np.arange(jmin, jmax + 2) * vs + origin[1]
    ze = np.arange(kmin, kmax + 2) * vs + origin[2]
    X, Y, Z = np.meshgrid(xe, ye, ze, indexing="ij")  # (ni+1, nj+1, nk+1)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(X, Y, Z, filled, edgecolor='k')  # draws cubes
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def visualize_points_and_voxels_open3d(
    frames_map, vox, cam_centers_map=None,
    max_points=200_000,
    max_voxels=30_000,
    point_size=2.0,
    voxel_wireframe=False,
    color_by="z",          # "z" | "hash" | "uniform"
    voxel_gap_ratio=0.08,  # fraction of voxel size removed on each edge (0..0.4)
    edge_overlay=True,     # draw thin LineSet edges around voxels
):
    """
    frames_map: list of (N_i,3) MAP-frame points (np.float32)
    cam_centers_map: list of (3,) MAP-frame camera centers
    vox: SparseVoxelGrid (same MAP frame)

    color_by:
        "z"     -> color ramp by world Z (blue->cyan->yellow)
        "hash"  -> pseudo-random stable color per IJK
        "uniform" -> single bluish color
    voxel_gap_ratio: 0 disables gaps, 0.05–0.15 is usually good.
    """
    import numpy as np, open3d as o3d

    def _hash_color(ijk):
        # deterministic pseudo-random color in [0,1]
        i, j, k = np.asarray(ijk, dtype=np.int64)
        h = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791)
        r = ((h >>  0) & 255) / 255.0
        g = ((h >>  8) & 255) / 255.0
        b = ((h >> 16) & 255) / 255.0
        # brighten a bit
        return 0.15 + 0.85 * np.array([r, g, b])

    def _z_color(z, zmin, zmax):
        # simple 3-stop ramp: blue -> cyan -> yellow
        if zmax <= zmin:
            t = 0.5
        else:
            t = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
        # piecewise: [0,0.5]: blue->cyan, [0.5,1]: cyan->yellow
        if t < 0.5:
            u = t / 0.5
            return np.array([0.0*(1-u) + 0.0*u, 1.0*u, 1.0])
        else:
            u = (t - 0.5) / 0.5
            return np.array([0.0*(1-u) + 1.0*u, 1.0, 1.0*(1-u) + 0.0*u])

    geoms = []

    # 1) point cloud (light gray)
    P = []
    for P_i in frames_map:
        if P_i.size:
            P.append(P_i[~np.isnan(P_i).any(axis=1)])
    if P:
        P = np.concatenate(P, axis=0).reshape(-1, 3)
        if P.shape[0] > max_points:
            idx = np.random.choice(P.shape[0], max_points, replace=False)
            P = P[idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
        col = np.full((P.shape[0], 3), 0.82, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(col)
        geoms.append(pcd)

    # 2) voxels as shrunken boxes with colors
    occ = vox.occupied_voxels()
    if len(occ) > max_voxels:
        occ = [occ[i] for i in np.random.choice(len(occ), max_voxels, replace=False)]

    if len(occ) > 0:
        vs = float(vox.p.voxel_size)
        shrink = max(0.0, min(0.4, voxel_gap_ratio))   # keep sane
        scale = 1.0 - shrink                           # uniform shrink factor
        big_mesh = o3d.geometry.TriangleMesh()
        edge_lines = []   # for LineSet overlay
        edge_pts = []

        # precompute z range for coloring
        if color_by == "z":
            centers = np.array([vox.ijk_to_center(ijk).astype(np.float64) for ijk in occ])
            zmin, zmax = float(centers[:, 2].min()), float(centers[:, 2].max())

        for ijk in occ:
            c = vox.ijk_to_center(ijk).astype(np.float64)

            box = o3d.geometry.TriangleMesh.create_box(width=vs, height=vs, depth=vs)
            box.translate(c - vs/2.0)
            # shrink around center to create visual gaps between neighbors
            box.scale(scale, center=c)

            # per-voxel color
            if color_by == "hash":
                col = _hash_color(ijk)
            elif color_by == "z":
                col = _z_color(c[2], zmin, zmax)
            else:
                col = np.array([0.2, 0.6, 1.0])  # uniform bluish

            box.paint_uniform_color(col.tolist())
            box.compute_vertex_normals()
            big_mesh += box

            if edge_overlay or voxel_wireframe:
                # collect 12 edges of the box for a LineSet overlay
                # corners of an axis-aligned box are available after translate/scale
                v = np.asarray(box.vertices)
                # index the 12 edges (pairs of vertex indices for open3d's create_box layout)
                edges = [(0,1),(1,3),(2,3),(0,2), (4,5),(5,7),(6,7),(4,6), (0,4),(1,5),(2,6),(3,7)]
                base = len(edge_pts)
                edge_pts.extend(v.tolist())
                edge_lines.extend([(base+a, base+b) for a,b in edges])

        geoms.append(big_mesh)

        if edge_overlay or voxel_wireframe:
            if edge_pts:
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(np.array(edge_pts)),
                    lines=o3d.utility.Vector2iVector(np.array(edge_lines, dtype=np.int32)),
                )
                # darker edges so voxel boundaries pop
                ls.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[0.05, 0.05, 0.05]]), (len(edge_lines), 1))
                )
                geoms.append(ls)

    # 3) optional camera centers
    if cam_centers_map:
        for C in cam_centers_map:
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=float(vox.p.voxel_size)*0.25)
            sp.translate(np.asarray(C, dtype=np.float64))
            sp.paint_uniform_color([1.0, 0.2, 0.2])
            geoms.append(sp)

    # 4) origin axes
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0]))

    # Use a Visualizer to tweak render options (smoother points, back faces, etc.)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Points + Voxels (MAP frame)", width=1280, height=800)
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True
    opt.light_on = True
    opt.background_color = np.array([1, 1, 1]) * 0.98  # very light gray helps edges pop

    vis.run()
    vis.destroy_window()
