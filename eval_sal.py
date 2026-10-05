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

from data.saliency_dataset_V2 import EventSaliencyDataset 
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



from evaluation.attention_helpers import *


cudnn.benchmark = False
#cudnn.deterministic = True
import torch
#torch.manual_seed(1000)
#torch.cuda.manual_seed(1000)


def visualize_batch(event, target, pred, opt,
                    max_samples: int = 3,
                    num_event_frames: int = 3,
                    num_output_frames: int = 3):
    """
    Visualizes each sample with:
    - Event input (multiple frames)
    - Last K target frames
    - Last K predicted frames
    """

    import numpy as np
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import random

    # -------------------------------------------------------
    # Helper: normalize shape → (B, T, C, H, W)
    # -------------------------------------------------------
    def ensure_B_T_C_H_W(x):
        """
        Normalizes shape so we always get (B, T, C, H, W).

        Supported input shapes:
        - (B, C, T, H, W)
        - (B, T, C, H, W)
        """
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor, got {x.shape}")

        B, a, b, H, W = x.shape

        # Case 1: (B, C, T, H, W)  → permute to (B, T, C, H, W)
        if a <= 3 and b > a:
            return x.permute(0, 2, 1, 3, 4)

        # Case 2: already (B, T, C, H, W)
        return x

    # -------------------------------------------------------
    # Move to CPU & normalize shapes
    # -------------------------------------------------------
    event = event.detach().float().cpu()   # (B,T,C,H,W) -- already correct
    target = ensure_B_T_C_H_W(target).detach().float().cpu()
    pred   = ensure_B_T_C_H_W(torch.sigmoid(pred)).detach().float().cpu()

    num_samples = min(max_samples, event.size(0))
    vis_h, vis_w = vis_size = (224, 224)

    # -------------------------------------------------------
    # Safe numpy conversion with normalization
    # -------------------------------------------------------
    def to_numpy_image_normalized(t):
        arr = t.cpu().numpy() if torch.is_tensor(t) else t
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    # -------------------------------------------------------
    # EVENT GRIDS (multiple random frames)
    # -------------------------------------------------------
    event_grids_list = []

    if event.ndim == 5:  # (B, T, C, H, W)
        B, T, C, H, W = event.shape

        if T <= num_event_frames:
            frame_idx = list(range(T))
        else:
            frame_idx = random.sample(range(T), num_event_frames)

        for i in range(num_samples):
            frames = event[i, frame_idx]  # (K,C,H,W)
            if C > 1:
                frames = frames.abs().max(dim=1, keepdim=True).values  # → (K,1,H,W)

            frames = F.interpolate(frames, size=vis_size, mode="bilinear", align_corners=False)

            frame_list = [frames[k, 0] for k in range(frames.size(0))]
            event_grids_list.append(torch.cat(frame_list, dim=1))

    else:
        event_grids_list = [
            torch.zeros((vis_h, vis_w * num_event_frames))
            for _ in range(num_samples)
        ]

    # -------------------------------------------------------
    # LAST K TARGET / PRED FRAMES
    # target/pred now: (B, T, C, H, W)
    # -------------------------------------------------------
    B, Tt, C, Ht, Wt = target.shape
    K = min(num_output_frames, Tt)

    # -------------------------------------------------------
    # PLOT
    # -------------------------------------------------------
    fig, axes = plt.subplots(num_samples, 3,
                             figsize=(8 + K * 3, 4 * num_samples))

    if num_samples == 1:
        axes = np.array([axes])

    for i in range(num_samples):
        ax_event, ax_target, ax_pred = axes[i]

        # ---------------- EVENT GRID ----------------
        ax_event.imshow(to_numpy_image_normalized(event_grids_list[i]), cmap="viridis")
        ax_event.set_title(f"Event input\n{num_event_frames} frames")
        ax_event.axis("off")

        # ---------------- TARGET K FRAMES ----------------
        tgt_frames = []
        for k in range(K):
            frame = target[i, -K + k, 0]       # (H,W)
            frame = frame.unsqueeze(0).unsqueeze(0)
            frame = F.interpolate(frame, size=vis_size, mode="bilinear", align_corners=False)
            tgt_frames.append(to_numpy_image_normalized(frame[0, 0]))

        ax_target.imshow(np.concatenate(tgt_frames, axis=1), cmap="gray")
        ax_target.set_title(f"Target (last {K} frames)")
        ax_target.axis("off")

        # ---------------- PRED K FRAMES ----------------
        pred_frames = []
        for k in range(K):
            frame = pred[i, -K + k, 0]
            frame = frame.unsqueeze(0).unsqueeze(0)
            frame = F.interpolate(frame, size=vis_size, mode="bilinear", align_corners=False)
            pred_frames.append(to_numpy_image_normalized(frame[0, 0]))

        ax_pred.imshow(np.concatenate(pred_frames, axis=1), cmap="gray")
        ax_pred.set_title(f"Prediction (last {K} frames)")
        ax_pred.axis("off")

    plt.tight_layout()
    save_path = f"./vis_UCF-001.png"
    plt.savefig(save_path)
    plt.close(fig)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, opt = update_opt(args.opt,args)    

    backbone = create_model(opt["network"],opt['logger']['name'])

    
    if args.other_model == 'angelo':
        model = AttentionModule()
    
    else:
        model = SaliencyEncoderDecoder(
                backbone,
                opt,
                )

        checkpoint = torch.load(args.checkpoint, map_location="cpu")#['model']

        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k.replace("model.", "", 1)] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print(device)
        model.to(device)
        model.eval()
    
    dataset_opt = deepcopy(opt["datasets"])

    val = dataset_opt.get('test_dir')
    event_bin = dataset_opt.get('event_bins')
    use_polarity = dataset_opt.get('use_polarity')
       
    val_dataset = EventSaliencyDataset( root_dir=val, event_bins=event_bin, use_polarity=use_polarity)

    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=20, num_workers=4, drop_last=False, pin_memory=True, persistent_workers = True, shuffle=False)

    plt = pl.Trainer(
        num_nodes=args.num_nodes, 
        precision = opt.get("precision", 32), 
        devices=args.gpus, 
    )
    
    # 4. Run Training
    print(args.opt,args.checkpoint)

    os.makedirs("best_samples", exist_ok=True)

    scores = defaultdict(list)

    best = {
        "sim": {"score": -1e9, "id": None, "pred": None},
        "cc": {"score": -1e9, "id": None, "pred": None},
        "nss": {"score": -1e9, "id": None, "pred": None},
        "auc": {"score": -1e9, "id": None, "pred": None},
    }


    evaluator = SaliencyEvaluator_sample()

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataset):
            event, label, fix, path = batch

            # 1. ALIGN DIMENSIONS
            # Label/Fix are [B, 1, T, H, W]. Move T to the second position: [B, T, 1, H, W]
            label = label.permute(0, 2, 1, 3, 4).to(device)
            fix = fix.permute(0, 2, 1, 3, 4).to(device)
            event = event.to(device)

            preds = model(event)  # [B, T, 1, H, W]
            B, T = preds.size(0), preds.size(1)

            if batch_idx == 0: # Usually you only want to visualize the first batch
                # opt = {"save_path": "./visualizations"} # Ensure this directory exists
                visualize_batch(event, label, preds, opt)

            for b in range(B):
                for t in range(T):
                    # 2. EXTRACT CORRESPONDING FRAMES
                    pred_bt = preds[b, t]  # Shape: [1, 224, 224]
                    sal_bt  = label[b, t]  # Shape: [1, 224, 224] - Now matches!
                    fix_bt  = fix[b, t]    # Shape: [1, 224, 224]
                    #print(pred_bt.shape, sal_bt.shape, fix_bt.shape)

                    sample_id = f"{batch_idx}_{b}_t{t}"

                    # 3. EVALUATE
                    res = evaluator.evaluate(pred_bt, sal_bt, fix_bt)

                    scores["sim"].append(res["SIM"])
                    scores["nss"].append(res["NSS"])
                    scores["cc"].append(res["CC"])
                    scores["auc"].append(res["AUC-J"])

                    if res["SIM"] > best["sim"]["score"]:
                        best["sim"] = {"score": res["SIM"], "id": sample_id, "pred": pred_bt.cpu()}

                    # Check NSS
                    if res["NSS"] > best["nss"]["score"]:
                        best["nss"] = {"score": res["NSS"], "id": sample_id, "pred": pred_bt.cpu()}

                    # Check CC
                    if res["CC"] > best["cc"]["score"]:
                        best["cc"] = {"score": res["CC"], "id": sample_id, "pred": pred_bt.cpu()}

                    # Check AUC
                    if res["AUC-J"] > best["auc"]["score"]:
                        best["auc"] = {"score": res["AUC-J"], "id": sample_id, "pred": pred_bt.cpu()}

    mean_sim = sum(scores["sim"]) / len(scores["sim"])
    mean_nss = sum(scores["nss"]) / len(scores["nss"])
    mean_cc = sum(scores["cc"]) / len(scores["cc"])
    mean_auc = sum(scores["auc"]) / len(scores["auc"])

    print(f"SIM: {mean_sim:.4f}")
    print(f"NSS: {mean_nss:.4f}")
    print(f"cc: {mean_cc:.4f}")
    print(f"auc: {mean_auc:.4f}")




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
    
