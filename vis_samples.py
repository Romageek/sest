#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings('ignore')
import argparse
from model import create_model
import torch.backends.cudnn as cudnn
import lightning.pytorch as pl
from trainer import create_trainer
from lightning.pytorch.strategies import DDPStrategy
from data import create_dataloader
from utils import MessageLogger, init_loggers, update_opt

from data.saliency_dataset import EventSaliencyDataset 
import wandb


from copy import deepcopy
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os 
from model.encoder_decoder_saliency import SaliencyEncoderDecoder

from metrics.evaluator import SaliencyEvaluator_sample
from collections import defaultdict

import cv2

from evaluation.attention_helpers import *


cudnn.benchmark = False
cudnn.deterministic = True
import torch
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)


def to_normalized_numpy(t):
    """ Matches the logic in your visualize_batch helper """
    arr = t.detach().cpu().numpy() if torch.is_tensor(t) else t
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, opt = update_opt(args.opt, args)    

    # 1. Load Model
    backbone = create_model(opt["network"], opt['logger']['name'])
    model = SaliencyEncoderDecoder(backbone, opt)
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device).eval()

    # 2. Setup Dataset (No DataLoader needed for specific samples)
    dataset_opt = deepcopy(opt["datasets"])
    val_dataset = EventSaliencyDataset(
        root_dir=dataset_opt.get('val_dir'), 
        event_bins=dataset_opt.get('event_bins'), 
        use_polarity=dataset_opt.get('use_polarity')
    )

    # 3. Define the specific indices you want to extract
    # You can change these to any indices from your validation set
    selected_indices = [i for i in range(47)] 
    output_root = "targeted_outputs"
    os.makedirs(output_root, exist_ok=True)

    print(f"Starting targeted inference on indices: {selected_indices}")

    with torch.no_grad():
        for idx in selected_indices:
            # 1. Load data
            event, label, fix, path = val_dataset[idx] 
            T_orig, C, H_orig, W_orig = event.shape
            
            # 2. Prepare for model (Resize to 224x224 while keeping T bins)
            # We want shape [1, T, C, 224, 224] for most temporal models
            event_resized = event.unsqueeze(0).to(device)
             
            preds = torch.sigmoid(model(event_resized))
            T_preds = preds.size(1)

            # Setup directories
            sample_dir = os.path.join(output_root, f"sample_{idx}")
            event_dir = os.path.join(sample_dir, "events")
            pred_dir = os.path.join(sample_dir, "predictions")
            os.makedirs(event_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)

            # 3. Iterate through time bins
            # Use event_resized[0] to get [T, C, H, W]
            vis_events = event_resized[0].cpu().numpy() 

            for t in range(T_preds):
                # --- A. PROCESS EVENT POLARITY ---
                # Indexing: [time, channel, height, width]
                pos = vis_events[t, 0] # Positive polarity
                neg = vis_events[t, 1] # Negative polarity
                
                event_rgb = np.zeros((224, 224, 3))
                event_rgb[:, :, 0] = to_normalized_numpy(pos) # Red
                event_rgb[:, :, 2] = to_normalized_numpy(neg) # Blue
                
                plt.imsave(os.path.join(event_dir, f"event_{t:03d}.png"), event_rgb)

                # --- B. PROCESS PREDICTION ---
                pred_frame = preds[0, t, 0].cpu().numpy()
                pred_vis = to_normalized_numpy(pred_frame)
                
                plt.imsave(os.path.join(pred_dir, f"pred_{t:03d}.png"), pred_vis, cmap='gray')

                # --- C. COMBINED PLOT ---
                fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor='black')
                axes[0].imshow(event_rgb)
                axes[0].set_title("Events (Red:+, Blue:-)", color='white')
                axes[0].axis('off')
                
                axes[1].imshow(pred_vis, cmap='gray', vmin=0, vmax=1)
                axes[1].set_title("Prediction", color='white')
                axes[1].axis('off')
                
                plt.savefig(os.path.join(sample_dir, f"combined_{t:03d}.png"), 
                            facecolor='white', bbox_inches='tight')
                plt.close(fig)

            print(f"Sample {idx} complete: {T_preds} frames saved. for {path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--gpus", default = 1, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--num_nodes", default = 1, type = int)
    parser.add_argument("--checkpoint", default = None, type = str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--opt', type=str, default = "", help='Path to option YAML file.')
    parser.add_argument('--other_model', type=str, default='no')
    
    args = parser.parse_args()
    main(args)
    