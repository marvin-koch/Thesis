"""
Keep synced timestamps across multiple camera folders (cam1..camK) with a stride,
writing results to out/camX/<timestamp>.jpg.

Assumptions
-----------
- Input root contains subfolders cam1, cam2, ..., camK.
- Each cam folder contains images named with a zero-padded numeric index + extension,
  e.g., 00000000.jpg, 00000001.jpg, etc. (png/jpeg also supported via --ext).
- "Synced" means we only keep timestamps that exist in *all* cameras (set intersection).

Output layout
-------------
out_root/
  cam1/
    00000000.jpg
    00000010.jpg
    ...
  cam2/
    00000000.jpg
    00000010.jpg
    ...

Usage
-----
python stride_sync_per_camera.py \
  --in-root /path/to/data \
  --out-root /path/to/out \
  --stride 10

Options:
  --ext .jpg                 # extension to filter (default .jpg)
  --cams cam1 cam2 cam3      # explicitly list camera names (default: auto-detect cam*)
  --copy                     # copy files (default)
  --move                     # move files instead of copy
  --symlink                  # create symlinks instead of copy
  --start 0                  # first index to include (inclusive)
  --end 99999999             # last index to include (inclusive)
  --dry-run                  # print actions only
  --pad 8                    # zero-padding width for filenames (default 8)

Example
-------
python stride_sync_per_camera.py --in-root ./dataset --out-root ./out --stride 5 --symlink
"""

import argparse
import re
from pathlib import Path
import shutil
from typing import Dict, List, Set

INDEX_RE = re.compile(r'(\d+)\.(?:jpg|jpeg|png)$', re.IGNORECASE)

def parse_args():
    p = argparse.ArgumentParser(description="Stride synced frames across cams into per-camera output folders.")
    p.add_argument("--in-root", required=True, type=Path, help="Root containing cam subfolders (cam1, cam2, ...)")
    p.add_argument("--out-root", required=True, type=Path, help="Output root (will create cam subfolders)")
    p.add_argument("--stride", required=True, type=int, help="Keep every Nth common timestamp")
    p.add_argument("--ext", default=".jpg", help="Image extension to filter (e.g., .jpg, .png)")
    p.add_argument("--cams", nargs="*", default=None, help="Camera folder names (default: detect 'cam*')")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--copy", action="store_true", help="Copy files (default)")
    g.add_argument("--move", action="store_true", help="Move files instead of copy")
    g.add_argument("--symlink", action="store_true", help="Symlink files instead of copy")
    p.add_argument("--start", type=int, default=None, help="First index to include (inclusive)")
    p.add_argument("--end", type=int, default=None, help="Last index to include (inclusive)")
    p.add_argument("--dry_run", action="store_true", help="Print actions; don't write")
    p.add_argument("--pad", type=int, default=8, help="Zero-padding for filenames in output")
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

    # Compute common indices across all cams (sync)
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

    def ensure_dir(p: Path):
        if args.dry_run:
            return
        p.mkdir(parents=True, exist_ok=True)

    def place(src: Path, dst: Path):
        if args.dry_run:
            print(f"[DRY] -> {dst}  (from {src})")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.symlink:
            try:
                rel = src.relative_to(dst.parent)
            except ValueError:
                rel = src
            if dst.exists():
                dst.unlink()
            dst.symlink_to(rel)
        elif args.move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)

    # Prepare output camera folders
    for cam in cams:
        ensure_dir(out_root / cam)

    # Write out per-camera files at kept indices
    for idx in kept:
        filename = f"{idx:0{args.pad}d}{ext.lower()}"
        for cam in cams:
            src = cam_maps[cam][idx]
            dst = out_root / cam / filename
            place(src, dst)

    print("Done.")

if __name__ == "__main__":
    main()
