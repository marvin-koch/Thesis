#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".png"}


def has_image(filenames):
    """Return True if any filename looks like an image."""
    return any(Path(f).suffix.lower() in IMG_EXTS for f in filenames)


def derive_seq_id(seq_dir):
    """
    Reproduce your seq_id logic:
        basis = os.path.basename(os.path.dirname(p)).replace(".basis", "")
        final = os.path.basename(p)
        seq_id = f"{basis}_{final}"
    """
    p = seq_dir.rstrip("/")
    basis = os.path.basename(os.path.dirname(p))
    basis = basis.replace(".basis", "")
    final = os.path.basename(p)
    return f"{basis}_{final}"


def build_manifest(dataset_root, check_gt=False):
    """
    Scans dataset_root once and returns a list of dict entries:
        { "seq_id": ..., "seq_path": ..., "has_gt": bool }
    """
    dataset_root = os.path.normpath(dataset_root)
    gt_root = os.path.join(dataset_root, "gt_voxels_per_timestep_01_v2")
    manifest = {}
    count = 0

    print(f"[INFO] Scanning dataset root: {dataset_root}")

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        base = os.path.basename(dirpath)

        # Only consider dirs named like "time0", "time1", ...
        if not base.startswith("time"):
            continue

        if not has_image(filenames):
            continue

        # The parent of the time-folder is the sequence folder
        seq_dir = os.path.dirname(dirpath)

        seq_id = derive_seq_id(seq_dir)

        # Avoid duplicates if multiple time folders point to the same seq
        if seq_id not in manifest:
            entry = {
                "seq_id": seq_id,
                "seq_path": seq_dir,
                "has_gt": False,
            }

            if check_gt:
                # Check if this sequence has t0000 GT
                gt_path_t0 = os.path.join(gt_root, f"{seq_id}_t0000_gt.npz")
                entry["has_gt"] = os.path.exists(gt_path_t0)

            manifest[seq_id] = entry
            count += 1

            if count % 100 == 0:
                print(f"  → Found {count} sequences so far...")

    print(f"[DONE] Total sequences discovered: {len(manifest)}")
    return list(manifest.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Dataset root containing scene folders")
    parser.add_argument("--out", required=True,
                        help="Where to write seq_manifest.json")
    parser.add_argument("--check-gt", action="store_true",
                        help="Whether to test for GT voxel files per sequence")
    args = parser.parse_args()

    manifest = build_manifest(args.root, check_gt=args.check_gt)

    print(f"[INFO] Saving manifest → {args.out}")
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SUCCESS] Manifest written with {len(manifest)} sequences.")


if __name__ == "__main__":
    main()
