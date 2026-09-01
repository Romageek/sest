import torch
import time
import datetime
import numpy as np
from utils import MetricLogger
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import BaseTrainer
import os 
import matplotlib.pyplot as plt
import torchvision
import random 
from metrics.evaluator import SaliencyEvaluator


def visualize_batch4D(event, target, pred, opt, max_samples: int = 3, num_event_frames: int = 3):

    #Visualizes 3 random event timeframes, ground truth target, and prediction side-by-side
    #for multiple batch samples, using a 1x3 subplot grid per sample.
    

    
    event = event.detach().to(torch.float32).cpu()
    target = target.detach().to(torch.float32).cpu()
    pred = torch.sigmoid(pred.detach().to(torch.float32).cpu())

    num_samples = min(max_samples, event.size(0))
    vis_size = (1, 224, 224) 
    
    event_batch = event[:num_samples]
    target_batch = target[:num_samples]
    pred_batch = pred[:num_samples]

    # Helper function for manual normalization (kept the same)
    def to_numpy_image_normalized(grid: torch.Tensor) -> np.ndarray:
        if grid.numel() == 0:
            return np.zeros(vis_size)
        g_min = grid.min()
        g_max = grid.max()
        if g_max > g_min:
            normalized_grid = (grid - g_min) / (g_max - g_min)
        else:
            normalized_grid = torch.zeros_like(grid)
        return normalized_grid.numpy()


    # --- 2. Temporal Sampling and Event Grid Creation ---
    
    event_grids_list = [] # Stores one torch tensor (H, W*N_frames) per sample
    
    if event_batch.ndim == 5: # (B, T, C, H, W)
        B, T, C, H, W = event_batch.shape
        
        if T < num_event_frames:
            frame_indices = list(range(T))
        else:
            frame_indices = random.sample(range(T), num_event_frames)
        
        for i in range(num_samples):
            # 1. Select the random frames for the current sample: (N_frames, C, H, W)
            sample_frames = event_batch[i, frame_indices, :, :, :] 
            
            # 2. Channel Aggregation (Max Abs Polarity): (N_frames, 1, H, W)
            if C > 1:
                sample_frames = sample_frames.abs().max(dim=1, keepdim=True).values 
            
            # 3. Resize: (N_frames, 1, vis_size[0], vis_size[1])
            sample_frames = F.interpolate(sample_frames, size=(224, 224), mode='bilinear', align_corners=False)
            
            # 4. Stack horizontally: (H, W * N_frames)
            # Use torch.cat on the list of (H, W) tensors
            event_grid_sample = torch.cat([f.squeeze() for f in sample_frames.chunk(sample_frames.size(0), dim=0)], dim=1)
            event_grids_list.append(event_grid_sample)

    else: # Fallback to black image if shape is unexpected or T is missing
        event_grids_list = [torch.zeros((vis_size[1], vis_size[2] * num_event_frames)) for _ in range(num_samples)]
        
    

    # Create a single figure with a row for each sample (or stick to one row for simplicity)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15 * num_event_frames / 3, 4 * num_samples))
    
    # If num_samples=1, axes is a 1D array. Normalize to 2D for consistent indexing.
    if num_samples == 1:
        axes = np.array([axes])
        
    # 1. Ensure Target is 4D (B, C, H, W)
    if target_batch.ndim == 5:
        target_to_resize = target_batch[:, -1] # Take last frame if video
    else:
        target_to_resize = target_batch # Already an image

    # 2. Ensure Prediction is 4D (B, C, H, W)
    if pred_batch.ndim == 5:
        pred_to_resize = pred_batch[:, -1] # Take last frame if video
    else:
        pred_to_resize = pred_batch # Already an image

    # 3. Use explicit size (224, 224) instead of vis_size tuple
    # Bilinear mode strictly requires (B, C, H, W)
    target_vis = F.interpolate(target_to_resize, size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
    pred_vis = F.interpolate(pred_to_resize, size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)


    for i in range(num_samples):
        row_axes = axes[i]
        
        # Plot 1: Event Input (multiple frames stacked)
        event_grid_numpy = to_numpy_image_normalized(event_grids_list[i])
        row_axes[0].imshow(event_grid_numpy, cmap='gray')
        row_axes[0].set_title(f"Input (Event) Sample {i+1}\n({num_event_frames} Random Frames)")
        
        # Plot 2: Target (GT)
        target_numpy = to_numpy_image_normalized(target_vis[i])
        row_axes[1].imshow(target_numpy, cmap='gray')
        row_axes[1].set_title(f"Target (GT) Sample {i+1}")
        
        # Plot 3: Prediction
        pred_numpy = to_numpy_image_normalized(pred_vis[i])
        row_axes[2].imshow(pred_numpy, cmap='gray')
        row_axes[2].set_title(f"Prediction Sample {i+1}")

        for ax in row_axes:
            ax.axis("off")

    plt.tight_layout()
    save_path = f"{opt['save_path']}/vis_UCF-001.png"
    plt.savefig(save_path)
    plt.close(fig)



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
    save_path = f"{opt['save_path']}/vis_UCF-001.png"
    plt.savefig(save_path)
    plt.close(fig)



class TrainerSaliency(BaseTrainer):
    
    def __init__(self, model, log, opt, checkpoint):
        super().__init__()
        
        self.model = model
        self.msg_logger = log
        self.opt = opt
        self.checkpoint = checkpoint
        self.automatic_optimization = True
        self.start_time  = None
        self.past_epoch = 0 

        self.criterion = SaliencyLoss(lambda_kl=1., lambda_cc=0.5, lambda_nss=0.5)
        self.evaluator = SaliencyEvaluator()


        os.makedirs(opt["save_path"],exist_ok=True)  

        if self.checkpoint != None:
            checkpoint = torch.load(self.checkpoint,map_location="cpu")
            self.past_epoch = checkpoint["current_epoch"]
            del checkpoint
            
        
    def training_step(self, batch, batch_idx):
        event, label, fix, _= batch
        target = label.float()
        
        pred = self.model(event)
        loss = self.criterion(pred, target, fix)
        visualize_batch(event, target, pred, self.opt)

      
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        event, target, fix, _= batch  # fixation optional
        pred = self.model(event)
        loss = self.criterion(pred, target, fix)

        #metrics = self.evaluator.evaluate(pred, target, None)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        visualize_batch(event, target, pred, self.opt)


    def test_step(self, batch, batch_idx):
        event, target, fix, path= batch  
        pred = self.model(event)
        loss = self.criterion(pred, target, fix)

        self.log("test_loss", loss, on_epoch=True)


        pred = pred.permute(0, 2, 1, 3, 4)

        metrics = self.evaluator.evaluate(pred, target, fix)
        for k, v in metrics.items():
            self.log(f"test_/{k}", v, prog_bar=True, on_epoch=True)
        
        visualize_batch(event, target, pred, self.opt)
    


        

    def configure_optimizers(self):
        
        self.opt["lr"] = (self.trainer.num_devices * self.trainer.num_nodes * self.num_batch_size / 256) * self.opt["base_lr"]
         
        optimizer = torch.optim.AdamW(
                                    self.model.parameters(),
                                    lr = self.opt["lr"],
                                    weight_decay = self.opt["weight_decay"]
                                    )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=1
        )
        
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
  

        if self.checkpoint != None:
            checkpoint = torch.load(self.checkpoint,map_location="cpu")
            key = "model" if "model" in checkpoint else "checkpoint"
            msg = self.model.load_state_dict(checkpoint[key],strict=False)
            if "optimizer" in checkpoint:
                 optimizer.load_state_dict(checkpoint["optimizer"])
            del checkpoint
        
        #self.optimizer = optimizer 
        #self.scheduler = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                'interval': 'epoch',
                'frequency': 1
            }
          
        }
    

class SaliencyLoss(nn.Module):
    def __init__(self, lambda_kl=1., lambda_cc=1.0, lambda_nss=1.0, eps=1e-8):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_cc = lambda_cc
        self.lambda_nss = lambda_nss
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def _flatten_btchw(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B*T, C, H, W)
        return x

    def kl_div(self, pred_logits, sal):
        pred = torch.sigmoid(pred_logits)

        P = pred / (pred.sum(dim=(-2,-1), keepdim=True) + self.eps)
        Q = sal  / (sal.sum(dim=(-2,-1), keepdim=True) + self.eps)

        KL = Q * (torch.log(Q + self.eps) - torch.log(P + self.eps))
        return KL.sum(dim=(-2,-1)).mean()

    def cc(self, pred_logits, sal):
        pred = torch.sigmoid(pred_logits)

        pred_n = (pred - pred.mean(dim=(-2,-1), keepdim=True)) / (pred.std(dim=(-2,-1), keepdim=True) + self.eps)
        sal_n  = (sal  - sal.mean(dim=(-2,-1), keepdim=True)) / (sal.std(dim=(-2,-1), keepdim=True) + self.eps)

        return (pred_n * sal_n).mean()

    def nss(self, pred_logits, fix):
        pred = torch.sigmoid(pred_logits)

        fix = (fix > 0).float()

        pred_n = (pred - pred.mean(dim=(-2,-1), keepdim=True)) / (pred.std(dim=(-2,-1), keepdim=True) + self.eps)

        return ((pred_n * fix).sum(dim=(-2,-1)) / (fix.sum(dim=(-2,-1)) + self.eps)).mean()

    def forward(self, pred_logits, saliency_gt, fixation_gt):
        pred_logits = pred_logits.permute(0, 2, 1, 3, 4)
        pred_logits = self._flatten_btchw(pred_logits)
        saliency_gt  = self._flatten_btchw(saliency_gt)
        fixation_gt  = self._flatten_btchw(fixation_gt)


        L_bce = self.bce(pred_logits, saliency_gt)
        
        L_kl  = self.kl_div(pred_logits, saliency_gt)
        L_cc  = self.cc(pred_logits, saliency_gt)
        L_nss = self.nss(pred_logits, fixation_gt)


        return 0.7*L_bce + self.lambda_kl * L_kl - self.lambda_cc * L_cc #- self.lambda_nss * L_nss #0.7*L_bce
 