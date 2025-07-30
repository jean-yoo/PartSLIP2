#!/usr/bin/env python3
"""
2D Proposal Evaluation Script
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
from pytorch3d.io import IO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'GLIP'))

from src.render_pc import render_single_view, render_pc 
from src.glip_inference import glip_inference, load_model, load_img
from src.utils import normalize_pc_from_mesh, load_colored_pc
from segment_anything import sam_model_registry, SamPredictor
from pytorch3d.structures import Pointclouds


class TwoD_ProposalEvaluator:
    def __init__(self, data_dir="./data", output_dir="./2d_eval_results", 
                 glip_config_file="./GLIP/configs/glip_Swin_L.yaml",
                 glip_model_path="./models/glip_large_model.pth",
                 sam_checkpoint="./models/sam_vit_h_4b8939.pth"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        # Load part names from metadata
        with open("PartNet_meta.json", "r") as f:
            self.part_meta = json.load(f)
        self.chair_parts = self.part_meta["chair"]
        
        # Initialize models
        print("[Loading GLIP model...]")
        self.glip_demo = load_model(glip_config_file, glip_model_path)
        
        print("[Loading SAM model...]")
        SAM_ENCODER_VERSION = "vit_h"
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        
        # Standard views from PartSLIP (elevation, azimuth)
        self.views = [[10, 0], [10, 90], [10, 180], [10, 270], 
                     [40, 0], [40, 120], [40, 240], 
                     [-20, 60], [-20, 180], [-20, 300]]
        
        self.rendering_styles = {
        }
        
        self.results = defaultdict(lambda: defaultdict(dict))  # style -> chair -> results
    
    def load_point_cloud(self, chair_id):
        """Load and normalize point cloud for a chair"""
        try:
            pc_path = self.data_dir / "test" / "Chair" / str(chair_id) / "pc.ply"
            if pc_path.exists():
                # Use PartSLIP's normalization function
                xyz, rgb = normalize_pc_from_mesh(
                    pc_file=str(pc_path), 
                    save_dir=None,  # Don't save intermediate files
                    device=self.device
                )
                return xyz, rgb
            else:
                print(f"Warning: No point cloud found for chair {chair_id}")
                return None, None
        except Exception as e:
            print(f"Error loading point cloud for chair {chair_id}: {e}")
            return None, None
    
    def render_with_style(self, xyz, rgb, style_params, save_dir, gt_semantic=None):
        """Render point cloud using standard PartSLIP rendering"""
        img_dir, pc_idx, screen_coords, num_views = render_pc(xyz, rgb, save_dir, self.device)
        
        return img_dir, pc_idx, screen_coords
    
    def run_glip_inference_style(self, save_dir, img_dir):
        """Run GLIP inference using PartSLIP's exact function"""
        # Use PartSLIP's glip_inference function directly
        masks = glip_inference(
            glip_demo=self.glip_demo,
            save_dir=save_dir,
            img_dir=img_dir,  # This should point to the rendered_img directory
            part_names=self.chair_parts,
            sam_predictor=self.sam_predictor,
            num_views=len(self.views),
            save_pred_img=True,
            save_individual_img=False
        )
        return masks

    def run_evaluation(self, chair_ids=None, max_chairs=None):
        """Run the simplified evaluation pipeline: for each rendering style -> for each chair -> save GLIP predictions"""
        print("Starting 2D Proposal Evaluation...")
        print(f"Rendering styles: {list(self.rendering_styles.keys())}")
        print(f"Views per rendering: {len(self.views)}")
        
        # Get list of chairs to evaluate
        if chair_ids is None:
            chair_dir = self.data_dir / "test" / "Chair"
            if chair_dir.exists():
                chair_ids = [d.name for d in chair_dir.iterdir() 
                           if d.is_dir() and d.name.isdigit()]
                chair_ids.sort()  # Consistent ordering
            else:
                print(f"Error: Chair directory {chair_dir} not found")
                return None
        
        if max_chairs:
            chair_ids = chair_ids[:max_chairs]
        
        print(f"Evaluating {len(chair_ids)} chairs across {len(self.rendering_styles)} styles...")
        
        # Main evaluation loop: style -> chair -> generate predictions
        for style_name, style_params in self.rendering_styles.items():
            for chair_id in tqdm(chair_ids, desc=f"Processing chairs for {style_name}"):
                try:
                    print(f"\n  Chair {chair_id}:")
                    
                    # Load chair data
                    xyz, rgb = self.load_point_cloud(chair_id)
                    if xyz is None or rgb is None:
                        print(f"    Skipping - could not load point cloud")
                        continue
                    
                    # Create save directory for this style/chair combination
                    save_dir = self.output_dir / f"style_{style_name}" / f"chair_{chair_id}"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Render with current style
                    print(f"    Rendering with style parameters...")
                    img_dir, pc_idx, screen_coords = self.render_with_style(
                        xyz, rgb, style_params, str(save_dir), gt_semantic=None  # No GT needed for simple version
                    )
                    
                    # Run GLIP inference across all views - this saves the prediction images
                    print(f"    Running GLIP inference across {len(self.views)} views...")
                    masks = self.run_glip_inference_style(str(save_dir), str(save_dir))
                    
                    print(f"     Saved rendered images to: {save_dir}/rendered_img/")
                    print(f"     Saved GLIP predictions to: {save_dir}/glip_pred/")
                    print(f"     Generated {len(self.views)} views with predictions")
                    
                except Exception as e:
                    print(f"    Error processing chair {chair_id} with style {style_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        print(f"All results saved to: {self.output_dir}")
        print(f"Structure: style_[name]/chair_[id]/rendered_img/ and glip_pred/")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="PartSLIP 2D Proposal Evaluation")
    parser.add_argument("--data-dir", default="./data", help="Path to data directory")
    parser.add_argument("--output-dir", default="./2d_eval_results", help="Output directory")
    parser.add_argument("--max-chairs", type=int, help="Maximum number of chairs to evaluate")
    parser.add_argument("--chair-ids", nargs="+", help="Specific chair IDs to evaluate")
    parser.add_argument("--glip-config", default="./GLIP/configs/glip_Swin_L.yaml", 
                       help="GLIP config file")
    parser.add_argument("--glip-model", default="./models/glip_large_model.pth", 
                       help="GLIP model path")
    parser.add_argument("--sam-checkpoint", default="./models/sam_vit_h_4b8939.pth",
                       help="SAM model checkpoint")
    
    args = parser.parse_args()
    
    evaluator = TwoD_ProposalEvaluator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        glip_config_file=args.glip_config,
        glip_model_path=args.glip_model,
        sam_checkpoint=args.sam_checkpoint
    )
    
    results = evaluator.run_evaluation(
        chair_ids=args.chair_ids,
        max_chairs=args.max_chairs
    )
    
    return results


if __name__ == "__main__":
    main() 