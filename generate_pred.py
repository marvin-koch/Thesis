import torch
import os
import numpy as np
from tqdm import tqdm
from train import HabitatSeqDataset
import pow3r2.tools.path_to_dust3r
from dust3r.model import AsymmetricCroCo3DStereo
from inference.utils import get_reconstructed_scene_no_opt
from preprocess_images.filter_images import changed_images
from torch.utils.data import Dataset, DataLoader
from dust3r.utils.image import load_images as li


from typing import Dict, List, Tuple, Optional
import os, shutil, json

import os, re, random

import PIL

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Config
SEQ_LIST = "/cluster/scratch/kochmar/renders/seq_manifest.json"
DATA_ROOT = "/cluster/scratch/kochmar/renders/"
SAVE_ROOT = "/cluster/scratch/kochmar/renders/precomputed_cache/"
WEIGHTS_PATH = "/cluster/home/kochmar/Thesis/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
STEP = 20

POINTS = "world_points"
CONF = "world_points_conf"


# ---------- utils ----------
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def _list_dirs(path: str) -> List[str]:
    return sorted(
        [os.path.join(path, d) for d in os.listdir(path)
         if os.path.isdir(os.path.join(path, d))],
        key=lambda p: _natural_key(os.path.basename(p))
    )

def _list_imgs(path: str, exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}) -> List[str]:
    files = [os.path.join(path, f) for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f))
             and os.path.splitext(f)[1].lower() in exts]
    files.sort(key=_natural_key)
    return files



import os
from typing import List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _has_image(filenames):
    return any(os.path.splitext(f)[1].lower() in IMG_EXTS for f in filenames)


def _sequence_dirs_from_root(root: str) -> List[str]:
    """
    For layouts like:

        root/
          scene1/.../sub_scene_X/
            time0/
              img.*
            time1/
              img.*
          scene2/.../sub_scene_Y/
            time0/
              img.*

    A 'sequence' is defined as the *parent directory* of any `time*` folder
    that actually contains images. There can be many sub-scenes and arbitrary
    nesting above them.
    """
    root = os.path.normpath(root)
    sequence_dirs = set()

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = os.path.normpath(dirpath)
        base = os.path.basename(dirpath)

        # We only care about timestep folders like time0, time1, ...
        if base.startswith("time"):
            if _has_image(filenames):      # or skip this check if you fully trust structure
                parent = os.path.dirname(dirpath)
                sequence_dirs.add(parent)

    if not sequence_dirs:
        raise FileNotFoundError(f"No sequences found under: {root}")

    # Natural sort by basename: scene1_sub1, scene1_sub2, ..., scene10_sub3
    seqs = sorted(sequence_dirs, key=lambda p: _natural_key(os.path.basename(p)))
    return seqs

# ---------- dataset ----------
class HabitatSeqDataset(Dataset):
    """
    Each sample = one sequence with timesteps.
    """
    def __init__(
        self,
        dataset_root: str,
        size: int = 512,
        verbose: bool = False,
        min_images_per_timestep: int = 1,
        sequences: Optional[List[str]] = None,   # pass a subset for train/val if you want
        skip=False,
        seq_list: str = "/cluster/scratch/kochmar/renders/seq_manifest.json"

    ):
        self.root = dataset_root
        self.size = size
        self.verbose = verbose
        self.min_images_per_timestep = min_images_per_timestep
        self.skip = skip
        self.seq_list = seq_list
        
        if sequences is None:
            # seqs = _sequence_dirs_from_root(dataset_root)
            with open(self.seq_list) as f:
                all_entries = json.load(f)

            if self.skip:
                seqs = [e["seq_path"] for e in all_entries if e["has_gt"]]
                all_ids  = [e["seq_id"]  for e in all_entries if e["has_gt"]]
            else:
                seqs = [e["seq_path"] for e in all_entries]
                all_ids  = [e["seq_id"]  for e in all_entries]
                
        else:
            seqs = [p if os.path.isabs(p) else os.path.join(dataset_root, p) for p in sequences]
        for s in seqs:
            if not os.path.isdir(s):
                raise FileNotFoundError(f"Sequence dir missing: {s}")
            
            
        self.seq_paths = seqs

    def __len__(self): return len(self.seq_paths)

    def _list_timesteps(self, seq_dir: str) -> List[str]:
        # timesteps are immediate subfolders; if none, treat the seq_dir itself as one timestep
        all_dirs = _list_dirs(seq_dir)
        valid_dirs = []
        for d in all_dirs:
            base = os.path.basename(d)
            if base.startswith("time") or base.isdigit():
                valid_dirs.append(d)

        # If no subfolders found, maybe the seq_dir itself is the timestep?
        return valid_dirs if valid_dirs else [seq_dir]

    def _load_timestep(self, t_dir: str) -> List[Dict]:
        img_paths = _list_imgs(t_dir)
        if len(img_paths) < self.min_images_per_timestep:
            return []

        try:
            # Attempt to load images using dust3r utils
            return li(img_paths, size=self.size, verbose=False)
        except (PIL.UnidentifiedImageError, OSError, Exception) as e:
            # --- CRASH PROTECTION ---
            # If a file is corrupt (e.g., cam_9.jpg), print a warning and SKIP this timestep.
            print(f"\n[WARN] Corrupted data in {t_dir}")
            print(f"       Skipping this timestep. Error: {e}")
            return []

    def __getitem__(self, idx: int) -> Dict:
        seq_dir = self.seq_paths[idx]
        p = seq_dir.rstrip("/") 
        basis = os.path.basename(os.path.dirname(p)) # "kfPV7w3FaU5.basis" 
        basis = basis.replace(".basis", "") # "kfPV7w3FaU5" 
        final = os.path.basename(p) # "0" 
        seq_id = f"{basis}_{final}"



        full_seq_dir = os.path.join(SAVE_ROOT, seq_id)
        save_path = os.path.join(full_seq_dir, f"t0000.pt")
        print(save_path)
        if os.path.exists(save_path):
            print("Skip get item, return None")
            #PUT THIS BACK
            return None

        t_dirs = self._list_timesteps(seq_dir)

        imgs_t: List[List[Dict]] = []
        for t, td in enumerate(t_dirs):
            if t % STEP != 0 and self.skip:
                continue
            imgs = self._load_timestep(td)
            if imgs:
                imgs_t.append(imgs)

        if not imgs_t:
            raise RuntimeError(f"No images found for sequence: {seq_dir}")

        return {
            "seq_id": seq_id,
            "seq_path": seq_dir,
            "timesteps": len(imgs_t),
            "imgs_t": imgs_t,   # List[List[dict]]; each inner list is what your inference() expects
        }
        


def check_feature_map_integrity(feature_map_tensor):
    """
    feature_map_tensor: (D, H, W) torch tensor (e.g., 64, 512, 512)
    """
    # 1. Reshape to (H*W, D)
    D, H, W = feature_map_tensor.shape
    feats = feature_map_tensor.permute(1, 2, 0).reshape(-1, D).cpu().numpy()

    # 2. PCA to 3 components (RGB)
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(feats)

    # 3. Normalize to 0-1 for display
    pca_feats = (pca_feats - pca_feats.min()) / (pca_feats.max() - pca_feats.min())
    pca_img = pca_feats.reshape(H, W, 3)

    # 4. Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(pca_img)
    plt.title(f"Feature Map PCA (If this looks like noise, extracting is broken)")
    plt.axis('off')
    plt.savefig("debug_feature_integrity.png")
    print("Saved debug_feature_integrity.png")

def precompute():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the heavy model once
    model = AsymmetricCroCo3DStereo.from_pretrained(WEIGHTS_PATH)
    model.to(device)
    model.eval()
    
    # 2. Setup Data
    dataset = HabitatSeqDataset(
        dataset_root=DATA_ROOT,
        size=512,
        verbose=False,
        seq_list  = SEQ_LIST
    )
    seqs = dataset.seq_paths
    
    keyrenders = []

    print(f"Starting precomputation for {len(seqs)} sequences...")


    #for seq_idx in (range(len(seqs))):
    #for seq_idx in (range(len(seqs)-1, -1, -1)):
    for seq_idx in (range(126, -1, -1)):

        print(seq_idx)
        batch = dataset[seq_idx]        # __getitem__ returns dict with seq info
        if batch is None:
            print("skip, already exists, batch is None")
            continue
        seq_id = batch["seq_id"]
        T = batch["timesteps"]

        # Create folder for this sequence
        seq_dir = os.path.join(SAVE_ROOT, seq_id)
        os.makedirs(seq_dir, exist_ok=True)
        mst = False

        for i in range(T):
            if i % STEP != 0:
                continue

            save_path = os.path.join(seq_dir, f"t{i:04d}.pt")
            if os.path.exists(save_path) and i != 0:
                print("Skip, already exists")
                #PUT THIS BACK
                continue

            imgs = batch["imgs_t"][i]
                    
        
            image_tensors = torch.stack([d["img"] for d in imgs])

            image_tensors = []
            for d in imgs:
                t = d["img"]                        # (1,3,H,W), likely float in [0,1]
                if t.ndim == 4 and t.shape[0] == 1:
                    t = t[0]                        # -> (3,H,W)
                t = t.detach().cpu()
                if not t.dtype.is_floating_point:
                    t = t.float()
                if t.max() > 1.0:                   # in case values are 0..255
                    t = t / 255.0
                image_tensors.append(t.clamp(0,1))
                
            image_tensors = torch.stack(image_tensors, dim=0)  
            image_tensors = image_tensors.to(device)

            if i < 1:
                

                predictions = get_reconstructed_scene_no_opt(i, ".", imgs, model, device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, projector=None)

                keyrenders = image_tensors.clone()
                                    

            else:
            

                changed_idx = changed_images(image_tensors, keyrenders, thresh=0.000005)
                
                print(changed_idx)

            
                
                if len(changed_idx) < 2:
                        # Advance epoch so the pipeline’s temporal bookkeeping stays aligned
                        print("no changes, skip")        

                        continue

                changed_idx = [0] + [x for x in changed_idx if x != 0]
                
                index_map = {new: old for new, old in enumerate(changed_idx)}
                        
                idx_t = torch.tensor(changed_idx, device=device, dtype=torch.long)
                keyrenders.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))
                
                
                print("final changed idx:", changed_idx)

    
                print("inference pred")
                if not mst:
                    mst = True
                    predictions = get_reconstructed_scene_no_opt(1, ".", imgs, model, device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx, projector=None)
                        
                else:
                    predictions = get_reconstructed_scene_no_opt(i, ".", imgs, model, device, False, 512, "", "linear", 100, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx, projector=None)

            # Filter out heavy unneeded data before saving if necessary
            # But keep "view_feats", "world_points", "world_points_conf", "extrinsic", "intrinsic_K"
            
            # Move everything to CPU to save disk space/memory
            needed = {
                "images","extrinsic", POINTS, CONF, "view_feats"
            }
            for k in list(predictions.keys()):
                if k not in needed:
                    del predictions[k]  # drop unneeded heavy stuff early


            cpu_pred = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in predictions.items()}
            
            # If view_feats is a list of tensors, move them too
            if "view_feats" in cpu_pred and isinstance(cpu_pred["view_feats"], list):
                cpu_pred["view_feats"] = [f.detach().cpu() if f is not None else None for f in cpu_pred["view_feats"]]

            #f_map = predictions["view_feats"][0] # Get first feature map
            #print(f"DEBUG: Saving tensor of shape {f_map.shape}")
            #check_feature_map_integrity(f_map)
            torch.save(cpu_pred, save_path)

if __name__ == "__main__":
    precompute()
