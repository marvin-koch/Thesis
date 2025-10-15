

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
