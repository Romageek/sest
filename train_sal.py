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


cudnn.benchmark = False
cudnn.deterministic = True
import torch
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
torch.cuda.manual_seed_all(42)  


import torch.multiprocessing as mp

class SaliencyDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = deepcopy(opt)

    def setup(self, stage: str):
        opt = self.opt
        dataset_opt = deepcopy(opt["datasets"])

        train = dataset_opt.get('train_dir')
        val = dataset_opt.get('val_dir')
        test = dataset_opt.get('test_dir')
        event_bin = dataset_opt.get('event_bins')
        use_polarity = dataset_opt.get('use_polarity')
        if train is None and test is None:
             raise ValueError("The 'root_dir' parameter must be defined in the 'datasets' configuration.")

        # Create Training Dataset 
        self.train_dataset = EventSaliencyDataset(root_dir=train,
                event_bins=event_bin,
                use_polarity=use_polarity,
        )
        
        # Create Validation Dataset 
        self.val_dataset = EventSaliencyDataset( root_dir=val,
                event_bins=event_bin,
                use_polarity=use_polarity,
        )

        # Create Test Dataset 
        self.test_dataset =  EventSaliencyDataset( root_dir=test,
                event_bins=event_bin,
                use_polarity=use_polarity,
        )




    def train_dataloader(self):
        if hasattr(self, "train_loader"): return self.train_loader
        train_loader = create_dataloader(self.train_dataset, self.opt['datasets'], self.opt['logger']['name'])
        self.train_loader = train_loader
        return train_loader

    def val_dataloader(self):
        if hasattr(self, "eval_loader"): return self.eval_loader
        eval_loader = create_dataloader(self.val_dataset, self.opt['datasets'], self.opt['logger']['name'], ddp_sampler=False)
        self.eval_loader = eval_loader
        return eval_loader
    
    def test_dataloader(self):
        if hasattr(self, "test_loader"): return self.test_loader
        test_loader = create_dataloader(self.test_dataset, self.opt['datasets'], self.opt['logger']['name'], ddp_sampler=False)
        self.test_loader = test_loader
        return test_loader

def main(args):

    args, opt = update_opt(args.opt,args)
    if "torch_home" in opt:
        os.environ['TORCH_HOME'] = opt["torch_home"]
    
    init_loggers(opt)
    msg_logger = MessageLogger(opt)

    # 1. Create Model Components
    backbone = create_model(opt["network"],opt['logger']['name'])
    model = SaliencyEncoderDecoder(
            backbone,
            opt,
            )
    for name, _ in model.named_parameters():
        print(name)
    # 2. Create Trainer Module
    pl_model = create_trainer(opt['train']['type'], opt['logger']['name'], 
                              {"model": model, "log" : msg_logger, "opt" : opt["train"], "checkpoint": args.checkpoint})
    
    # 3. Setup Trainer Arguments
    kwargs = {}
    if opt.get("apex",False):
        kwargs = {"amp_backend":"apex", "amp_level":"O1"}
        
    # Use the modern DDPStrategy
    strategy = DDPStrategy(find_unused_parameters=True)
    
    sync_batchnorm = opt['train'].get('sync_batchnorm', True)
    check_val_every_n_epoch = opt['train'].get('check_val_every_n_epoch',1)

    # Define callbacks
    parts = [item.split('/')[-2] for item in opt['datasets']['train_dir']]
    datast_str = "_".join(parts)

    help_str =opt['network']['decoder']+'_decoder_'+datast_str + '_dataset_'+ str(opt['datasets']['event_bins'])+ 'frames_' + str(opt['head_main']['num_classes'])
    checkpoint_callback = ModelCheckpoint(
        dirpath=help_str,
        monitor="val_loss",
        filename="{cnn}-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=False
    )

    
    # OR WandB (if you're using wandb)
    wandb_logger = WandbLogger(project="saliency", name=help_str)

    
    plt = pl.Trainer(
        max_epochs = opt["train"].get("early_stop_epoch", opt["train"]["epoch"]) - pl_model.past_epoch,
        num_nodes=args.num_nodes, 
        callbacks=[checkpoint_callback, early_stopping],
        precision = opt.get("precision", 32), 
        devices=args.gpus, 
        strategy=strategy, 
        logger = wandb_logger, 
        profiler = SimpleProfiler(), 
        sync_batchnorm = sync_batchnorm, 
        check_val_every_n_epoch = check_val_every_n_epoch, 
        **kwargs
    )
    
    # 4. Run Training
    data_mod = SaliencyDataModule(opt)
    plt.fit(pl_model, datamodule=data_mod)
    plt.test(pl_model, datamodule=data_mod )
    

if __name__ == "__main__":
    import torch
    try:
        # Use 'spawn' to create clean CUDA contexts for worker processes
        torch.multiprocessing.set_start_method('spawn', force=True) 
    except RuntimeError:
        pass 
    mp.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser() 
    parser.add_argument("--gpus", default = 1, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--num_nodes", default = 1, type = int)
    parser.add_argument("--checkpoint", default = None, type = str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--opt', type=str, default = "", help='Path to option YAML file.')
    
    args = parser.parse_args()
    wandb.login()
    main(args)
    wandb.finish()