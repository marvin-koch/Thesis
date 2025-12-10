import torch
import os
import numpy as np
from tqdm import tqdm
from train import HabitatSeqDataset
import pow3r2.tools.path_to_dust3r
from dust3r.model import AsymmetricCroCo3DStereo
from inference.utils import get_reconstructed_scene_no_opt
from preprocess_images.filter_images import changed_images

# Config
SEQ_LIST = "/cluster/scratch/kochmar/renders/seq_manifest.json"
DATA_ROOT = "/cluster/scratch/kochmar/renders/"
SAVE_ROOT = "/cluster/scratch/kochmar/renders/precomputed_cache/"
WEIGHTS_PATH = "/cluster/home/kochmar/Thesis/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

POINTS = "world_points"
CONF = "world_points_conf"
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
    )
    seqs = dataset.seq_paths
    
    keyframes = []

    print(f"Starting precomputation for {len(seqs)} sequences...")


    for seq_idx in range(len(seqs)):
        batch = dataset[seq_idx]        # __getitem__ returns dict with seq info
        seq_id = batch["seq_id"]
        T = batch["timesteps"]

        # Create folder for this sequence
        seq_dir = os.path.join(SAVE_ROOT, seq_id)
        os.makedirs(seq_dir, exist_ok=True)
        mst = False

        for i in range(T):
            if i % 10 != 0:
                continue

            save_path = os.path.join(seq_dir, f"t{i:04d}.pt")
            if os.path.exists(save_path):
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
                

                predictions = get_reconstructed_scene_no_opt(i, ".", imgs, model, device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0, projector=None)

                keyframes = image_tensors.clone()
                                    

            else:
            

                changed_idx = changed_images(image_tensors, keyframes, thresh=0.000005)
                
                print(changed_idx)

            
                
                if len(changed_idx) < 2:
                        # Advance epoch so the pipeline’s temporal bookkeeping stays aligned
                        print("no changes, skip")        

                        continue

                changed_idx = [0] + [x for x in changed_idx if x != 0]
                
                index_map = {new: old for new, old in enumerate(changed_idx)}
                        
                idx_t = torch.tensor(changed_idx, device=device, dtype=torch.long)
                keyframes.index_copy_(0, idx_t, image_tensors.index_select(0, idx_t))
                
                
                print("final changed idx:", changed_idx)

    
                print("inference pred")
                if not mst:
                    mst = True
                    predictions = get_reconstructed_scene_no_opt(1, ".", imgs, model, device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx, projector=None)
                        
                else:
                    predictions = get_reconstructed_scene_no_opt(i, ".", imgs, model, device, False, 512, "", "linear", 50, 1, True, False, True, False, 0.05, "oneref", 1, 0, changed_gids=changed_idx, projector=None)

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

            torch.save(cpu_pred, save_path)

if __name__ == "__main__":
    precompute()