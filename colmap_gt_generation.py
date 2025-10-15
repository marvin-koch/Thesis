# pip install pycolmap
import pycolmap

import os, numpy as np, torch

def _camera_model_name(cam) -> str:
    # pycolmap 3.12.5: cam.model is an enum; in older, it may be a string/int
    m = getattr(cam, "model", None)
    if m is None:
        raise AttributeError("pycolmap.Camera has no 'model' attribute")
    if hasattr(m, "name"):
        return str(m.name).upper()
    return str(m).upper()

def _K_from_camera(cam) -> np.ndarray:
    name = _camera_model_name(cam)  # returns upper-case string
    # params order follows COLMAP conventions
    # docs: https://colmap.github.io/format.html#camera-models
    p = np.asarray(cam.params, dtype=np.float32)

    if name in (
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    ):
        # p = [f, cx, cy, ...]
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f

    elif name in (
        "PINHOLE",
        "OPENCV",
        "FULL_OPENCV",
        "OPENCV_FISHEYE",
        "FOV",  # p = [fx, fy, cx, cy, omega]
    ):
        # p = [fx, fy, cx, cy, ...]
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]

    else:
        raise ValueError(f"Unsupported COLMAP camera model: {name}")

    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def _qvec_to_R(q):
    # q = [qw,qx,qy,qz] → R (world→cam)
    qw, qx, qy, qz = q
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=np.float32)

import numpy as np

def image_Rt_world_to_cam(im) -> np.ndarray:
    """
    Returns [R|t] mapping world -> camera, robust across pycolmap versions.
    Works with pycolmap 3.12.5 where qvec/tvec are gone and cam_from_world() is callable.
    """
    # 1) get pose object (Rigid3d or similar)
    pose = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world

    # 2) try the explicit rotation/translation first
    R = None; t = None

    # rotation
    if hasattr(pose, "rotation"):
        rot = pose.rotation
        rot = rot() if callable(rot) else rot
        if hasattr(rot, "matrix"):
            Rm = rot.matrix
            Rm = Rm() if callable(Rm) else Rm
            R = np.asarray(Rm, dtype=np.float32)

    # translation
    if hasattr(pose, "translation"):
        tr = pose.translation
        tr = tr() if callable(tr) else tr
        t = np.asarray(tr, dtype=np.float32).reshape(3,1)

    # 3) fallback: some builds expose a full 4x4
    if (R is None or t is None) and hasattr(pose, "matrix"):
        M = pose.matrix
        M = M() if callable(M) else M
        M = np.asarray(M, dtype=np.float32)
        R = M[:3, :3]; t = M[:3, 3:4]

    if R is None or t is None:
        raise AttributeError("Could not extract rotation/translation from cam_from_world()")

    return np.concatenate([R, t], axis=1)

def _track_len(track) -> int:
    """Robustly get the number of observations in a pycolmap Track."""
    # Common accessors
    if hasattr(track, "length"):
        v = track.length
        return int(v() if callable(v) else v)
    if hasattr(track, "size"):
        v = track.size
        return int(v() if callable(v) else v)
    if hasattr(track, "elements"):
        elems = track.elements
        elems = elems() if callable(elems) else elems
        try:
            return int(len(elems))
        except TypeError:
            # elements is iterable but not sized
            return int(sum(1 for _ in elems))
    # Fallbacks: try generic len()/iteration
    try:
        return int(len(track))
    except TypeError:
        try:
            return int(sum(1 for _ in track))
        except TypeError:
            return 1  # last-resort: treat as single observation

def _db_stats(db_path):
    import sqlite3
    con = sqlite3.connect(db_path); cur = con.cursor()
    imgs = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    kpts = cur.execute("SELECT SUM(rows) FROM keypoints").fetchone()[0] or 0
    desc = cur.execute("SELECT SUM(rows) FROM descriptors").fetchone()[0] or 0
    pairs = cur.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
    print(f"[DB] images={imgs} keypoint_rows={kpts} descriptor_rows={desc} verified_pairs={pairs}")
    # Optional: show a few per-image keypoint counts
    rows = cur.execute("SELECT image_id, rows FROM keypoints LIMIT 5").fetchall()
    print("[DB] sample keypoints rows:", rows)
    con.close()

import sqlite3, textwrap

def _db_stats_2(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Detect inlier column name
    tvg_cols = [r[1] for r in cur.execute("PRAGMA table_info(two_view_geometries);")]
    inl_col = "num_inliers" if "num_inliers" in tvg_cols else "rows"

    # Basic counts
    n_imgs = cur.execute("SELECT COUNT(*) FROM images;").fetchone()[0]
    n_kps  = cur.execute("SELECT COALESCE(SUM(rows),0) FROM keypoints;").fetchone()[0]
    n_pairs= cur.execute("SELECT COUNT(*) FROM two_view_geometries;").fetchone()[0]
    n_inlier_pairs = cur.execute(
        f"SELECT COUNT(*) FROM two_view_geometries WHERE {inl_col} >= 15;"
    ).fetchone()[0]
    sum_inliers = cur.execute(
        f"SELECT COALESCE(SUM({inl_col}),0) FROM two_view_geometries;"
    ).fetchone()[0]

    # Top pairs by inliers (decode pair_id -> (id1,id2) via bit ops)
    top_pairs = cur.execute(f"""
        SELECT i1.name, i2.name, t.{inl_col}
        FROM two_view_geometries t
        JOIN images i1 ON i1.image_id = (t.pair_id >> 32)
        JOIN images i2 ON i2.image_id = (t.pair_id & 4294967295)
        ORDER BY t.{inl_col} DESC
        LIMIT 10;
    """).fetchall()

    # Graph degree per image (how many verified neighbors)
    degrees = cur.execute("""
        WITH edges AS (
            SELECT (pair_id >> 32) AS u, (pair_id & 4294967295) AS v
            FROM two_view_geometries
        ),
        undirected AS (
            SELECT u AS a, v AS b FROM edges
            UNION ALL
            SELECT v AS a, u AS b FROM edges
        )
        SELECT images.name, COUNT(*) AS degree
        FROM images
        LEFT JOIN undirected ON images.image_id = undirected.a
        GROUP BY images.image_id
        ORDER BY degree DESC, images.name ASC;
    """).fetchall()

    print(textwrap.dedent(f"""
      DB REPORT
      images:          {n_imgs}
      total keypoints: {n_kps}
      verified pairs:  {n_pairs}   (>=15 inliers: {n_inlier_pairs})
      sum inliers:     {sum_inliers}
      top pairs (img1, img2, inliers): {top_pairs}
      degrees (image, degree): {degrees[:10]}
    """))

    con.close()

    


# import os, shutil, subprocess, glob, pycolmap

# def ensure_sparse_model(seq_dir: str, overwrite: bool = False, overlap: int = 5) -> str:
#     """
#     Builds a COLMAP sparse model at {seq_dir}/colmap/sparse/<id> using CLI.
#     Returns the path to the first sparse model directory.
#     Raises a RuntimeError with captured logs if anything fails or 0 images are registered.
#     """
#     assert os.path.isdir(seq_dir), f"Not a directory: {seq_dir}"
#     imgs = sorted([p for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG")
#                    for p in glob.glob(os.path.join(seq_dir, ext))])
#     if not imgs:
#         raise RuntimeError(f"No images found in {seq_dir}")

#     col_dir    = os.path.join(seq_dir, "colmap")
#     db_path    = os.path.join(col_dir, "database.db")
#     sparse_dir = os.path.join(col_dir, "sparse")

#     if overwrite and os.path.isdir(col_dir):
#         shutil.rmtree(col_dir)
#     os.makedirs(sparse_dir, exist_ok=True)

#     def run(cmd):
#         proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
#         return proc.stdout, proc.stderr


#     # run(["colmap" ,"automatic_reconstructor",
#     #     "--workspace_path", col_dir,
#     #     "--image_path", seq_dir])
    
#     # 1) feature extraction
#     run([
#         "colmap","feature_extractor",
#         "--database_path", db_path,
#         "--image_path", seq_dir,
#         "--ImageReader.single_camera","1",
#         "--SiftExtraction.use_gpu","1",
#         "--SiftExtraction.max_num_features", "8000",
#         "--SiftExtraction.peak_threshold", "0.008",
#         "--SiftExtraction.first_octave=-1",
#         "--SiftExtraction.num_octaves", "4",
#         "--SiftExtraction.domain_size_pooling", "1",
#         # "--ImageReader.camera_model", "OPENCV",
#         # "--ImageReader.camera_params", "220.7922249468612,294.0824871346696,160.28496512299822,128.4565687581635,-0.08178931008842633,0.14104190015816495,-0.002007631490187864,-0.0010861073551153564"

#         # "--ImageReader.camera_model", "PINHOLE",
#         # "--ImageReader.camera_params", "217.02853451482747,289.0311142574589,159.859769561464,127.8833834555531",
#         # "--ImageReader.camera_params", "162.771,385.375,119.892,170.511"

#     ])
    

    
#     _db_stats(db_path)  # expect images>0 and keypoint_rows>0


#     # 2) sequential matching (good for videos / ordered frames)
#     # run([
#     #     "colmap","sequential_matcher",
#     #     "--database_path", db_path,
#     #     "--SequentialMatching.overlap", str(overlap),
#     #     "--SequentialMatching.quadratic_overlap","1"
#     # ])
    
#     run([
#         "colmap","exhaustive_matcher",
#         "--database_path", db_path,
#         "--SiftMatching.guided_matching","1",
#         "--SiftMatching.max_num_matches" ,"32768"

#     ])
    
#     _db_stats(db_path)  # expect images>0 and keypoint_rows>0
    
#     _db_stats_2(db_path)


#     # # 3) mapping
#     out_before = set(os.listdir(sparse_dir))
#     # run([
#     #     "colmap","mapper",
#     #     "--database_path", db_path,
#     #     "--image_path", seq_dir,
#     #     "--output_path", sparse_dir
     
#     # ])
    
#     run([
#         "colmap", "mapper",
#         "--database_path", db_path,
#         "--image_path", seq_dir,
#         "--output_path", sparse_dir,
#         "--Mapper.min_num_matches", "5",
#         "--Mapper.init_min_num_inliers", "6",
#         "--Mapper.abs_pose_min_num_inliers", "6",
#         "--Mapper.ba_refine_focal_length", "1",
#         "--Mapper.ba_refine_principal_point", "1",
#         "--Mapper.ba_refine_extra_params", "1",
#         "--Mapper.multiple_models", "0",
#         "--Mapper.tri_min_angle", "0.5",
#     ])

#     out_after = sorted(d for d in os.listdir(sparse_dir) if os.path.isdir(os.path.join(sparse_dir, d)))
#     new_models = [d for d in out_after if d not in out_before]
#     model_id = new_models[0] if new_models else (out_after[0] if out_after else None)
#     if model_id is None:
#         raise RuntimeError(f"Mapper produced no model under {sparse_dir}")

#     model_dir = os.path.join(sparse_dir, model_id)

#     # Sanity: check registered images
#     rec = pycolmap.Reconstruction(model_dir)
#     n_reg = len(rec.images)
    
#     if n_reg == 0:
#         raise RuntimeError(
#             "Sparse model has 0 registered images. Causes: poor matches, wrong seq_dir, or EXIF/camera issues.\n"
#             f"Model dir: {model_dir}"
#         )
#     return model_dir

import os as _os
_os.environ.setdefault("GLOG_minloglevel", "2")
_os.environ.setdefault("COLMAP_VERBOSE", "0")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # avoid MPS hard-crash
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # avoid MPS hard-crash

def load_colmap_predictions(seq_dir: str, device: str = "cuda"):
    """
    seq_dir: folder that contains images and a COLMAP sparse model at seq_dir/colmap/sparse/<model_id>
    Returns predictions dict with: intrinsic (S,3,3), extrinsic (S,3,4), world_points (N,3), world_points_conf (N,)
    """
    
    rec = ensure_sparse_model(seq_dir, overwrite=False, overlap=5, pairing="exhaustive")

    # #pick first sparse model dir
    # sparse_root = os.path.join(seq_dir, "colmap", "sparse")
    # assert os.path.isdir(sparse_root), f"Missing COLMAP sparse at: {sparse_root}"
    # sub = sorted([d for d in os.listdir(sparse_root) if os.path.isdir(os.path.join(sparse_root, d))])
    # assert sub, f"No models under {sparse_root}"
    

    # model_dir = os.path.join(sparse_root, sub[0])

    # rec = pycolmap.Reconstruction(out_dir)
    
    print("Registered images:", len(rec.images))
    print("Cameras:", len(rec.cameras))
    print("Points3D:", len(rec.points3D))

    # Order images by filename (consistent with your loops)
    images = sorted(rec.images.values(), key=lambda im: im.name)
    
    print(len(images))
    
    Rt_list = [image_Rt_world_to_cam(im) for im in images]

    print(Rt_list)

    K_list = []
    for im in images:
        cam = rec.cameras[im.camera_id]
        K_list.append(_K_from_camera(cam))
       

    # Sparse points (optional)
    if len(rec.points3D):
        pts = np.stack([np.asarray(P.xyz, dtype=np.float32) for P in rec.points3D.values()], axis=0)
        errs = np.array([float(P.error) for P in rec.points3D.values()], dtype=np.float32)
        trkl = np.array([max(1, _track_len(P.track)) for P in rec.points3D.values()], dtype=np.float32)
        conf = (1.0 / (1.0 + errs)) * np.minimum(1.0, trkl / 5.0)
    else:
        pts = np.zeros((0,3), dtype=np.float32)
        conf = np.zeros((0,), dtype=np.float32)

    predictions = {
        "intrinsic":  torch.from_numpy(np.stack(K_list)).to(device),
        "extrinsic":  torch.from_numpy(np.stack(Rt_list)).to(device),   # world->cam
        "world_points":      torch.from_numpy(pts).to(device),
        "world_points_conf": torch.from_numpy(conf).to(device),
    }
    return predictions




def ensure_sparse_model(seq_dir: str, overwrite: bool = False, overlap: int = 5,
                        use_hloc: bool = True,
                        pairing: str = "exhaustive",   # "exhaustive" or "sequential"
                        feature_conf: str = "superpoint_max",  # hloc extract_features.confs key
                        matcher_conf: str = "superpoint+lightglue"        # hloc match_features.confs key
                        ):
    """
    If use_hloc=True, builds a COLMAP sparse model with SuperPoint+LightGlue via hloc
    and writes the model to {seq_dir}/colmap/sparse/0 (cameras.bin, images.bin, points3D.bin).
    Otherwise, falls back to your original COLMAP-SIFT pipeline.
    Returns the path to the model dir that contains the three COLMAP .bin files.
    """
    import os, glob, shutil, subprocess, sqlite3, textwrap
    from pathlib import Path
    import pycolmap

    assert os.path.isdir(seq_dir), f"Not a directory: {seq_dir}"
    imgs = sorted([p for ext in ("*.jpg","*.jpeg","*.png","*.JPG","*.PNG")
                   for p in glob.glob(os.path.join(seq_dir, ext))])
    if not imgs:
        raise RuntimeError(f"No images found in {seq_dir}")

    col_dir    = os.path.join(seq_dir, "colmap")
    db_path    = os.path.join(col_dir, "database.db")
    sparse_dir = os.path.join(col_dir, "sparse")
    model_out  = os.path.join(sparse_dir, "0")  # we’ll write here for compatibility
    os.makedirs(sparse_dir, exist_ok=True)

    if overwrite and os.path.isdir(col_dir):
        shutil.rmtree(col_dir)
        os.makedirs(sparse_dir, exist_ok=True)

    if use_hloc:
        # ----- hloc route: SP + LightGlue + COLMAP mapping -----
        from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction, match_dense

        images = Path(seq_dir)
        out_root = Path(col_dir) / "hloc"
        out_root.mkdir(parents=True, exist_ok=True)


        import cv2

        src = images
        dst = images / "scaled"
        dst.mkdir(parents=True, exist_ok=True)

        target_long_edge = 1600  # or 1280 if you want smaller

        for img_path in src.glob("*.[jp][pn]g"):  # matches .jpg/.png
            img = cv2.imread(str(img_path))
            if img is None:
                print("Skipping", img_path)
                continue

            h, w = img.shape[:2]
            scale = target_long_edge / max(h, w)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))

            # Use high-quality interpolation for upscaling
            img_up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(str(dst / img_path.name), img_up)
            print(f"{img_path.name}: {w}x{h} -> {new_w}x{new_h}")

        # 1) feature extraction
        feat_cfg = extract_features.confs.get(feature_conf)
        if feat_cfg is None:
            raise ValueError(f"Unknown hloc feature_conf '{feature_conf}'. "
                             f"Available: {list(extract_features.confs.keys())}")
            
            
        # retrieval_conf = extract_features.confs["netvlad"]
        images = dst
        
        features_path = extract_features.main(feat_cfg, images, out_root)
        
        pairs_path = out_root / "pairs-exhaustive.txt"
        

        pairs_from_exhaustive.main(
                output=pairs_path,
                features=features_path,          # <-- use the HLOC features file
                # (optional) you could also pass ref_features=features_path for self-matching
        )
        
        # 3) match with LightGlue
       
        
        # matches_path = match_features.main(match_cfg, pairs_path, features_path, out_root)


        # matches_out = out_root / f"matches-{matcher_conf}.h5"

        # matches_path = match_features.main(
        #     conf=match_cfg,
        #     pairs=pairs_path,          # Path to pairs .txt
        #     features=features_path,    # Path to features .h5
        #     matches=matches_out        # Path to OUTPUT matches .h5  (not a directory)
        # )
        # features_path = extract_features.main(feat_cfg, images, out_root)


        match_cfg = match_features.confs.get(matcher_conf)
        print(match_cfg)
        match_cfg["model"]["weights"] = 'indoor'
    
        matches_path = match_features.main(match_cfg, pairs_path, feat_cfg["output"], out_root)
        
        
        
        # dense_conf = match_dense.confs['loftr']
        # dense_conf["model"]["weights"] = 'indoor'
        # dense_conf["max_error"] = 4
        # dense_conf["cell_size"] = 4

        # features, matches_path = match_dense.main(dense_conf, pairs_path, images, export_dir=out_root)#, features_ref=features_path)

        sfm_dir = (out_root / f"sfm_{feature_conf}+{matcher_conf}")
        
        # from hloc.triangulation import (
        #     OutputCapture,
        #     estimation_and_geometric_verification,
        #     import_features,
        #     import_matches,
        #     parse_option_args,
        # )
        
        # mapper_opts =[
        #     "init_min_num_inliers=25",
        #     "init_max_error=6.0",
        #     "init_min_tri_angle=4.0",
        #     "init_max_forward_motion=0.99",
        #     "init_max_reg_trials=6",
        #     "abs_pose_min_num_inliers=8",
        #     "abs_pose_min_inlier_ratio=0.10",
        #     "abs_pose_max_error=12.0",
        #     "abs_pose_refine_focal_length=False",
        #     "abs_pose_refine_extra_params=False",
        #     "filter_max_reproj_error=6.0",
        #     "filter_min_tri_angle=0.5",
        #     "local_ba_num_images=4",
        #     "local_ba_min_tri_angle=2.0",
        #     "max_reg_trials=6"
        # ]

        # model = reconstruction.main((sfm_dir), (images), pairs_path, features_path, matches_path, camera_mode=pycolmap.CameraMode.SINGLE) #,mapper_options=parse_option_args(mapper_opts, pycolmap.IncrementalMapperOptions()))

        # return model
        
        
        
        
        # # 4) COLMAP reconstruction via hloc (runs mapper under the hood)
        sfm_dir = out_root / f"sfm_{feature_conf}+{matcher_conf}"
        sfm_dir.mkdir(exist_ok=True, parents=True)

        # # mirror your mapper options
        mapper_opts = {
            "min_num_matches": 5,
            "init_min_num_inliers": 6,
            "abs_pose_min_num_inliers": 6,
            "ba_refine_focal_length": True,
            "ba_refine_principal_point": True,
            "ba_refine_extra_params": True,
            "multiple_models": False,
            "tri_min_angle": 0.5,
        }

        # camera_mode: "SINGLE" matches your ImageReader.single_camera=1
        # reconstruction.main(
        #     sfm_dir=sfm_dir,
        #     image_dir=images,
        #     pairs=pairs_path,
        #     features=features_path,
        #     matches=matches_path,
        #     camera_mode="SINGLE",
        #     mapper_options=mapper_opts
        # )
        
        
     
        # from pycolmap import CameraMode, ImageReaderOptions, IncrementalMapperOptions
        # img_opts = ImageReaderOptions()
        # map_opts = IncrementalMapperOptions()
        
        # map_opts.init_min_num_inliers = 6

        # #img_opts.single_camera = True                     # one shared camera
        # img_opts.camera_model = "RADIAL_FISHEYE"          # enforce the model
        
        # reconstruction.main(
        #     sfm_dir=sfm_dir,                        # Path ok
        #     image_dir=(images),                  # MUST be str
        #     pairs=pairs_path,
        #     features=features_path,
        #     matches=matches_path,
        #     camera_mode=CameraMode.SINGLE,          # MUST be enum
        #     image_options=img_opts,      # MUST be object, not dict
        #     #mapper_options=mapper_opts
            
        # )

      

        # # 5) copy to seq_dir/colmap/sparse/0 so the rest of your code works unchanged
        # for fn in ("cameras.bin", "images.bin", "points3D.bin"):
        #     src = sfm_dir / fn
        #     if not src.exists():
        #         raise RuntimeError(f"Expected file not found in hloc sfm dir: {src}")
        # os.makedirs(model_out, exist_ok=True)
        # for fn in ("cameras.bin", "images.bin", "points3D.bin"):
        #     shutil.copy2(sfm_dir / fn, os.path.join(model_out, fn))

        # # sanity check
        # rec = pycolmap.Reconstruction(model_out)
        # n_reg = len(rec.images)
        # if n_reg == 0:
        #     raise RuntimeError(
        #         "Sparse model has 0 registered images after hloc reconstruction. "
        #         "Possible causes: weak texture, too large frame spacing, or pairing config."
        #     )
        # return rec
    
    
   

        
        from hloc.reconstruction import create_empty_db, get_image_ids, import_features, import_matches
        import pycolmap
        from pycolmap import CameraMode, ImageReaderOptions

        # sfm_dir already defined above
        sfm_dir.mkdir(exist_ok=True, parents=True)
        database = sfm_dir / "database.db"
        create_empty_db(database)
        
        from pathlib import Path
        exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG",".PNG"}
        image_names = sorted([p.name for p in Path(seq_dir).iterdir() if p.suffix in exts])
        if not image_names:
            raise RuntimeError(f"No images found under {seq_dir}")

        # 1) Import images into COLMAP DB (use correct arg types)
        img_opts = ImageReaderOptions()  # tweak if needed
        
      
        #img_opts.single_camera = True                     # one shared camera
        img_opts.camera_model = "RADIAL_FISHEYE"          # enforce the model
        
        pycolmap.import_images(
            (database),               # str, NOT Path
            (images),                 # str, NOT Path
            camera_mode=CameraMode.SINGLE,
            image_names=image_names,              # all images in folder
            options=img_opts
        )

        # 2) Import features & matches into the DB (hloc helpers are fine here)
        
        image_ids = get_image_ids(database)
        import_features(image_ids, database, features_path)
        import_matches(image_ids, database, pairs_path, matches_path)

        # 3) Run COLMAP mapper CLI with your options
        def _run(cmd):
            import subprocess
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
            return p.stdout

        def _print_db_quick_report(dbfile: str):
            import sqlite3
            con = sqlite3.connect(dbfile); cur = con.cursor()
            imgs = cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            kpts = cur.execute("SELECT COALESCE(SUM(rows),0) FROM keypoints").fetchone()[0]
            desc = cur.execute("SELECT COALESCE(SUM(rows),0) FROM descriptors").fetchone()[0]
            pairs= cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0] \
                if cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='matches'").fetchone() \
                else 0
            print(f"[HL0C DB] images={imgs} keypoints_rows={kpts} descriptors_rows={desc} pairs_listed={pairs}")
            con.close()

        _print_db_quick_report(str(database))

       

        # Sanity: print verified edges and inliers
        import sqlite3
        con = sqlite3.connect(str(database)); cur = con.cursor()
        cols = [r[1] for r in cur.execute("PRAGMA table_info(two_view_geometries);")]
        inl_col = "num_inliers" if "num_inliers" in cols else "rows"
        n_pairs  = cur.execute("SELECT COUNT(*) FROM two_view_geometries;").fetchone()[0]
        sum_inl  = cur.execute(f"SELECT COALESCE(SUM({inl_col}),0) FROM two_view_geometries;").fetchone()[0]
        print(f"[DB] verified_pairs={n_pairs}, sum_inliers={sum_inl}")
        con.close()
        _run([
            "colmap", "mapper",
            "--database_path", str(database),
            "--image_path", str(images),
            "--output_path", str(sfm_dir),
            "--Mapper.min_num_matches", "1",
            "--Mapper.init_min_num_inliers", "1",
            "--Mapper.abs_pose_min_num_inliers", "1",
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "1",
            "--Mapper.ba_refine_extra_params", "1",
            "--Mapper.multiple_models", "0",
            "--Mapper.tri_min_angle", "0.5",
        ])

        # 4) Copy model to seq_dir/colmap/sparse/0 (so your downstream stays unchanged)
        for fn in ("cameras.bin", "images.bin", "points3D.bin"):
            src = sfm_dir / fn
            if not src.exists():
                raise RuntimeError(f"Expected file not found: {src}")

        os.makedirs(model_out, exist_ok=True)
        for fn in ("cameras.bin", "images.bin", "points3D.bin"):
            shutil.copy2(sfm_dir / fn, os.path.join(model_out, fn))

        # 5) Sanity check
        rec = pycolmap.Reconstruction(sfm_dir)
        if len(rec.images) == 0:
            raise RuntimeError("Sparse model has 0 registered images after mapping.")
        return rec


    # # ----- fallback: your original pure-COLMAP (SIFT) route -----
    # def run(cmd):
    #     proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #     if proc.returncode != 0:
    #         raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    #     return proc.stdout, proc.stderr

    # # 1) SIFT features
    # run([
    #     "colmap","feature_extractor",
    #     "--database_path", db_path,
    #     "--image_path", seq_dir,
    #     "--ImageReader.single_camera","1",
    #     "--SiftExtraction.use_gpu","1",
    #     "--SiftExtraction.max_num_features", "8000",
    #     "--SiftExtraction.peak_threshold", "0.008",
    #     "--SiftExtraction.first_octave=-1",
    #     "--SiftExtraction.num_octaves", "4",
    #     "--SiftExtraction.domain_size_pooling", "1",
    # ])

    # _db_stats(db_path)

    # # 2) matching (exhaustive)
    # run([
    #     "colmap","exhaustive_matcher",
    #     "--database_path", db_path,
    #     "--SiftMatching.guided_matching","1",
    #     "--SiftMatching.max_num_matches" ,"32768"
    # ])

    # _db_stats(db_path)
    # _db_stats_2(db_path)

    # # 3) mapping
    # out_before = set(os.listdir(sparse_dir))
    # run([
    #     "colmap", "mapper",
    #     "--database_path", db_path,
    #     "--image_path", seq_dir,
    #     "--output_path", sparse_dir,
    #     "--Mapper.min_num_matches", "5",
    #     "--Mapper.init_min_num_inliers", "6",
    #     "--Mapper.abs_pose_min_num_inliers", "6",
    #     "--Mapper.ba_refine_focal_length", "1",
    #     "--Mapper.ba_refine_principal_point", "1",
    #     "--Mapper.ba_refine_extra_params", "1",
    #     "--Mapper.multiple_models", "0",
    #     "--Mapper.tri_min_angle", "0.5",
    # ])

    # out_after = sorted(d for d in os.listdir(sparse_dir) if os.path.isdir(os.path.join(sparse_dir, d)))
    # new_models = [d for d in out_after if d not in out_before]
    # model_id = new_models[0] if new_models else (out_after[0] if out_after else None)
    # if model_id is None:
    #     raise RuntimeError(f"Mapper produced no model under {sparse_dir}")

    # model_dir = os.path.join(sparse_dir, model_id)
    # rec = pycolmap.Reconstruction(model_dir)
    # n_reg = len(rec.images)
    # if n_reg == 0:
    #     raise RuntimeError(
    #         "Sparse model has 0 registered images. Causes: poor matches, wrong seq_dir, or EXIF/camera issues.\n"
    #         f"Model dir: {model_dir}"
    #     )
    # return model_dir


import torch
from vggt.models.vggt import VGGT
import sys
import numpy as np
import time
from voxel.utils import *
from voxel.voxel import *
from voxel.align import *
from preprocess_images.filter_images import changed_images



if __name__ == "__main__":

    sys.path.append("vggt/")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    POINTS = "world_points"
    CONF = "world_points_conf"
    VIZ = True
    threshold = 62.0     
    z_clip_map = (-0.5, 0.3)   
    R_w2m = np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]], dtype=np.float32)
    t_w2m = np.zeros(3, dtype=np.float32)



    voxel_size = 0.01
    vox = TorchSparseVoxelGrid(
        origin_xyz=np.zeros(3, dtype=np.float32),
        params=VoxelParams(voxel_size=voxel_size, promote_hits=2),
        device=device, dtype=torch.float32
    )

    target_dir = "/Users/marvin/Documents/Thesis/vggt/examples/fishbowl1/"
    # sub_dirs = ["images","images", "images1", "images2", "images3"]
    # sub_dirs = ["00000000", "00000050","00000100", "00000150", "00000200", "00000250"]
    # sub_dirs = ["00000000", "00000300",  "00000350", "00000400"]

    sub_dirs = sorted([d for d in os.listdir(target_dir) 
                if os.path.isdir(os.path.join(target_dir, d))])


    for i, images in enumerate(sub_dirs):
        print(f"Iteration {i}")

        print(images)
        #run inference
        # image_tensors = load_images(target_dir + images, device=device)

        start = time.time()

        predictions = load_colmap_predictions(target_dir + images, device=device)

        
        end = time.time()
        length = end - start

        print("Running Colmap inference took", length, "seconds!")
        

        # Keep tensors; only extract what we need later.
        # If you truly need NumPy later, convert specific keys then.
        needed = {
            "images","extrinsic", "intrinsic", POINTS, CONF
        }
        
        for k in list(predictions.keys()):
            if k not in needed:
                del predictions[k]  # drop unneeded heavy stuff early


        # print("Align Point Cloud")

        start = time.time()
        
        WPTS_m = rotate_points(predictions[POINTS], R_w2m, t_w2m)
        Rmw, tmw, info = align_pointcloud(WPTS_m, inlier_dist=voxel_size*0.75)
        WPTS_m = rotate_points(WPTS_m, Rmw, tmw)
        predictions[POINTS] = WPTS_m

        # if VIZ:
        #     visualize_vggt_pointcloud(predictions, key=POINTS, conf_key=CONF, threshold=threshold)

        # print("Building Voxels and BEV")

        camera_R = R_w2m @ Rmw
        camera_t = t_w2m + tmw
        frames_map, cam_centers_map, conf_map, (S,H,W) = build_frames_and_centers_vectorized(
            predictions,
            POINTS=POINTS,
            CONF=CONF,
            threshold=threshold,
            Rmw=camera_R, tmw=camera_t,
            z_clip_map=z_clip_map,   # or None
        )   
        end = time.time()
        length = end - start

        print("Aligning and building frames/camera centers took", length, "seconds!")


        align_to_voxel = (i > 0)

        start = time.time()

        # if i == 0:
        #     vox.begin_bootstrap()
        vox, bev, meta = build_maps_from_points_and_centers_torch(
            frames_map,
            cam_centers_map,
            conf_map,
            vox,
            align_to_voxel=align_to_voxel,
            voxel_size=voxel_size,           # 10 cm
            bev_window_m=(10.0, 10.0), # local 20x20 m
            bev_origin_xy=(0.0, 0.0),
            z_clip_vox=(-np.inf, np.inf),
            z_band_bev=(0.05, 0.3),
            max_range_m=None,
            carve_free=True,
            samples_per_voxel=1,
            ray_stride=2
        )

        # if i == 0:
        #     vox.end_bootstrap()   # promotes current occupied to LT and clears ST
        #     vox.lock_long_term()  # optional: forbids any future LT change
            
        end = time.time()
        length = end - start

        print("Building Voxel and BEV took", length, "seconds!")

        # print("BEV:", bev.shape, meta)    

        num_cells = vox.keys.numel() if hasattr(vox, "keys") else len(vox.logodds)
        num_occ = int((vox.vals > vox.p.occ_thresh).sum().item()) if hasattr(vox, "vals") else \
                sum(1 for L in vox.logodds.values() if L > vox.p.occ_thresh)

        # print("Voxels (stored cells):", num_cells)
        # print("Voxels (occupied):", num_occ)

        # On-screen plots
        if VIZ:
            visualize_bev(bev, meta)
            # visualize_voxels(vox, z_band=(-0.5, 3), max_dim=200)
            visualize_points_and_voxels_open3d(frames_map, vox,cam_centers_map,
                                            max_points=150_000, max_voxels=25_000)


        save_root = "test_colmap/"
        # Files
        save_bev(bev, meta, save_root + f"bev_{i}.png", save_root + f"bev_{i}_np.npy", save_root + f"bev_{i}_meta.json")
        export_occupied_voxels(vox,save_root + f"voxels{i}.ply", save_root + f"voxels{i}_ijk.npy", save_root + f"voxels{i}_meta.json",z_clip_map)

        vox.next_epoch()
