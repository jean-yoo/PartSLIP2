import os
import torch
import json
from pytorch3d.io import IO
import numpy as np
from src.utils import normalize_pc, normalize_pc_from_mesh
from src.render_pc import render_pc
from src.gen_superpoint import gen_superpoint

def Infer(input_pc_file, category, part_names, zero_shot=False, save_dir="tmp"):
    
    print("[creating tmp dir...]")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    io = IO()
    os.makedirs(save_dir, exist_ok=True)
    
    print("[normalizing input point cloud...]")
    xyz, rgb = normalize_pc_from_mesh(pc_file =input_pc_file, save_dir = save_dir, device = device)
    
    print("[rendering input point cloud...]")
    img_dir, pc_idx, screen_coords, num_views = render_pc(xyz = xyz, rgb = rgb, save_dir =save_dir, device = device)
    
    # print('[generating superpoints...]')
    superpoint = gen_superpoint(xyz, rgb, visualize=True, save_dir=save_dir)
    
    print("[finish!]")
    
if __name__ == "__main__":
    partnet_meta = json.load(open("PartNet_meta.json")) 
    categories = list(partnet_meta.keys())
    # [["Box", "Bucket", "Clock", "CoffeeMachine"],
    #                    ["Dishwasher", "Eyeglasses", "Faucet", "FoldingChair"],
    #                    ["Lighter", "Microwave", "Mouse", "Pen", "WashingMachine"],
    #                     ["Phone", "Pliers", "Printer", "Refrigerator", "Window"],
    #                     ["Remote", "Safe", "Scissors", "Stapler"],
    #                     ["Switch", "Toilet", "TrashCan", "USB"]]

    # categories = ["Camera", "Cart", "Dispenser", "Kettle"]
    # categories = ["Bottle", "Chair", "Display", "Door"]
    # categories = ["Knife", "Lamp", "StorageFurniture", "Table"]
    # categories = ["KitchenPot", "Oven", "Suitcase", "Toaster"]
    # categories = categories_list
    for category in categories:  
        models = os.listdir(f"./data/partnet/{category}") # list of models
                # models = sorted(models)
        for model in models:
            model_path = os.path.join(f"./data/partnet/{category}", model)
            if model.startswith("._") or model == ".DS_Store":
                try:
                    if os.path.isdir(model_path):
                        import shutil
                        shutil.rmtree(model_path)
                        print(f"[Deleted resource fork directory: {model_path}]")
                    else:
                        os.remove(model_path)
                        print(f"[Deleted resource fork file: {model_path}]")
                except Exception as e:
                    print(f"[Failed to delete {model_path}: {e}]")
                continue  # Skip macOS resource fork files and .DS_Store
            # Only process if it's a directory
            # if not os.path.isdir(model_path):
            #     print(f"[Skipping non-directory: {model_path}]")
            #     continue
            Infer(f"./data/partnet/{category}/ut_vis_chair_37569.ply", category, partnet_meta[category], zero_shot=False, save_dir=f"./data/img_sp/{category}_partnet/{model}")
