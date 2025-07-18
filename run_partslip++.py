from src.mask2ins_refine import sem2ins
from pytorch3d.io import IO
import os
from src.utils import normalize_pc, normalize_pc_from_mesh
import torch
import numpy as np
import json

META_FILE = "PartNet_meta.json"
CAT_FOLDER = "chair_partnet"

def test(input_pc_file, part_names, sp_dir, save_dir="tmp"):
    io = IO()
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda:0")

    xyz, rgb = normalize_pc_from_mesh(pc_file =input_pc_file, save_dir = save_dir, device = device)

    idx_dir = f"{sp_dir}/idx_dir"
    pc_idx = np.load(f"{sp_dir}/idx.npy", allow_pickle=True)
    screen_coords = np.load(f"{sp_dir}/coor.npy", allow_pickle=True)

    sem2ins(xyz, rgb, screen_coords, pc_idx, part_names, 
                   save_dir, 20, pc_idx.shape[0], img_dir=sp_dir)
    
if __name__ == "__main__":
    partnete_meta = json.load(open(META_FILE)) 
    categories = partnete_meta.keys()
    #categories = ["Chair"]

    for category in categories:
        #models = os.listdir(f"./data/img_sp/{CAT_FOLDER}") # list of models
        models = ['ut_vis_chair_37569.ply']
        for model in models:
            if model.startswith("._"):
                model_path = os.path.join(f"./data/img_sp/{CAT_FOLDER}", model)
                try:
                    if os.path.isdir(model_path):
                        import shutil
                        shutil.rmtree(model_path)
                        print(f"[Deleted resource fork directory: {model_path}]")
                    else:
                        os.remove(model_path)
                        print(f"[Deleted resource fork file: {model_path}]")
                except Exception as e:
                    print(f"[Failed to delete {model_path}: {e}")
                continue  # Skip macOS resource fork files
            print(f"Category: {category}, Model: {model}")
            test(f"./data/partnet/{category}/{model}", partnete_meta[category], 
                 sp_dir=f"./data/img_sp/{CAT_FOLDER}/{model}",
                 save_dir=f"./result_ps++/{CAT_FOLDER}/{model}")