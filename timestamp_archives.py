#!/usr/bin/env python3
"""
Create ONE archive per timestamp (synced across cams, with stride),
each archive containing the images from all cameras for that timestamp.

Output layout (default: ZIP):
  out_root/
    00000000.zip    # contains cam1.jpg, cam2.jpg, ..., camK.jpg
    00000010.zip
    ...

Assumptions
-----------
- Input root contains subfolders cam1, cam2, ..., camK.
- Each cam folder contains images named as zero-padded numeric index + extension,
  e.g., 00000000.jpg, 00000001.jpg, etc. (png/jpeg also supported via --ext).
- "Synced" means we only keep timestamps that exist in *all* cameras (set intersection).

Usage
-----
python timestamp_archives.py \
  --in-root /path/to/data \
  --out-root /path/to/out \
  --stride 10

Options
-------
  --ext .jpg                 # extension to filter (default .jpg)
  --cams cam1 cam2 ...       # explicitly list camera names (default: auto-detect cam*)
  --start 0                  # first index to include (inclusive)
  --end 99999999             # last index to include (inclusive)
  --pad 8                    # zero-padding width for the archive filenames (default 8)
  --format zip|tar|folder    # output container type (default zip)
  --zip-store                # ZIP: store (no compression). Default is deflate compression.
  --dry-run                  # print actions without writing

Examples
--------
# ZIP archives with deflate compression (default), every 5th timestamp
python timestamp_archives.py --in-root ./dataset --out-root ./out --stride 5

# TAR archives (uncompressed)
python timestamp_archives.py --in-root ./dataset --out-root ./out --stride 10 --format tar

# Just make folders per timestamp instead of archives
python timestamp_archives.py --in-root ./dataset --out-root ./out --stride 10 --format folder
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set
import zipfile
import tarfile

INDEX_RE = re.compile(r'(\d+)\.(?:jpg|jpeg|png)$', re.IGNORECASE)

def parse_args():
    p = argparse.ArgumentParser(description="Create one archive per timestamp containing synced cam images, with stride.")
    p.add_argument("--in-root", required=True, type=Path, help="Root containing cam subfolders (cam1, cam2, ...)")
    p.add_argument("--out-root", required=True, type=Path, help="Where to write timestamp archives/folders")
    p.add_argument("--stride", required=True, type=int, help="Keep every Nth common timestamp")
    p.add_argument("--ext", default=".jpg", help="Image extension filter (e.g., .jpg, .png)")
    p.add_argument("--cams", nargs="*", default=None, help="Camera subfolder names (default: detect 'cam*')")
    p.add_argument("--start", type=int, default=None, help="First index to include (inclusive)")
    p.add_argument("--end", type=int, default=None, help="Last index to include (inclusive)")
    p.add_argument("--pad", type=int, default=8, help="Zero-padding for archive names")
    p.add_argument("--format", choices=["zip","tar","folder"], default="zip", help="Output container type")
    p.add_argument("--zip-store", action="store_true", help="ZIP: use store (no compression). Default: deflate compression")
    p.add_argument("--dry-run", action="store_true", help="Print actions; don't write")
    return p.parse_args()

def find_cameras(in_root: Path, cams_opt: List[str] | None) -> List[str]:
    if cams_opt:
        return cams_opt
    cams = sorted([p.name for p in in_root.iterdir() if p.is_dir() and p.name.lower().startswith("cam")])
    if not cams:
        raise FileNotFoundError(f"No camera folders found under {in_root}. Expected cam1, cam2, ...")
    return cams

def index_map_for_camera(cam_dir: Path, ext: str) -> Dict[int, Path]:
    ext_lower = ext.lower()
    mapping: Dict[int, Path] = {}
    for f in cam_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() != ext_lower:
            continue
        m = INDEX_RE.search(f.name)
        if not m:
            continue
        idx = int(m.group(1))
        mapping[idx] = f
    return mapping

def write_zip(out_path: Path, members: Dict[str, Path], compresslevel: int | None, store: bool, dry: bool):
    if dry:
        print(f"[DRY] ZIP -> {out_path} :: {list(members.keys())}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_STORED if store else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(out_path, mode="w", compression=compression, compresslevel=compresslevel) as zf:
        for arcname, src in members.items():
            zf.write(src, arcname=arcname)

def write_tar(out_path: Path, members: Dict[str, Path], dry: bool):
    if dry:
        print(f"[DRY] TAR -> {out_path} :: {list(members.keys())}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, mode="w") as tf:
        for arcname, src in members.items():
            tf.add(src, arcname=arcname)

def write_folder(out_dir: Path, members: Dict[str, Path], dry: bool):
    if dry:
        print(f"[DRY] FOLDER -> {out_dir} :: {list(members.keys())}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # We copy files to the folder. To keep this script dependency-free we just write hardlinks if possible,
    # otherwise we copy by opening/reading/writing.
    for name, src in members.items():
        dst = out_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dst.exists():
                dst.unlink()
            # try hardlink for speed/space
            dst.hardlink_to(src)
        except Exception:
            # fallback to copy
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())

def main():
    args = parse_args()
    in_root: Path = args.in_root
    out_root: Path = args.out_root
    stride: int = args.stride
    ext: str = args.ext

    cams = find_cameras(in_root, args.cams)
    print(f"Detected cameras: {cams}")

    # Build per-cam index maps
    cam_maps: Dict[str, Dict[int, Path]] = {}
    for cam in cams:
        cam_dir = in_root / cam
        if not cam_dir.exists():
            raise FileNotFoundError(f"Missing camera folder: {cam_dir}")
        cam_maps[cam] = index_map_for_camera(cam_dir, ext)

    # Intersect indices to enforce sync
    index_sets: List[Set[int]] = [set(m.keys()) for m in cam_maps.values()]
    if not index_sets:
        print("No images found.")
        return

    common = set.intersection(*index_sets)
    if args.start is not None:
        common = {i for i in common if i >= args.start}
    if args.end is not None:
        common = {i for i in common if i <= args.end}
    if not common:
        print("No common timestamps with current filters.")
        return

    sorted_idx = sorted(common)
    kept = sorted_idx[::stride]
    print(f"Common timestamps: {len(sorted_idx)}; keeping {len(kept)} with stride {stride}")

    # Prepare writing
    for idx in kept:
        ts_str = f"{idx:0{args.pad}d}"
        # Members to include in archive/folder: {"cam1.jpg": Path(...), ...}
        members: Dict[str, Path] = {}
        for cam in cams:
            src = cam_maps[cam][idx]
            # Name inside the archive/folder is camX + extension
            members[f"{cam}{ext.lower()}"] = src

        if args.format == "zip":
            out_path = out_root / f"{ts_str}.zip"
            # compresslevel default None lets Python choose. If store is True, no compression.
            write_zip(out_path, members, compresslevel=None, store=args.zip_store, dry=args.dry_run)
        elif args.format == "tar":
            out_path = out_root / f"{ts_str}.tar"
            write_tar(out_path, members, dry=args.dry_run)
        else:  # folder
            out_dir = out_root / ts_str
            write_folder(out_dir, members, dry=args.dry_run)

    print("Done.")

if __name__ == "__main__":
    main()
