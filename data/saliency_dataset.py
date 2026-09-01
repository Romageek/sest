import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from torchvision import transforms
from . import seg_utils as data_util
import random
import gc

import torch.nn.functional as F

class EventSaliencyDataset(data.Dataset):
    """
    Dataset class adapted MainDir/Sequence/ structure,
    pairing 'events/*.npz' with the *last* 'maps/*.png' as GT.
    """
    def __init__(self,
                 root_dir,  # Renamed event_dir to root_dir to encompass MainDir
                 event_folder="events",
                 saliency_folder="maps",
                 fixations_folder="fixation",
                 event_bins=5,
                 use_polarity=False,
                 train=True,
                 augmentation=False):
        super().__init__()
        self.root_dir = root_dir
        self.event_folder = event_folder
        self.saliency_folder = saliency_folder
        self.fixations_folder = fixations_folder
        self.event_bins = event_bins
        self.event_polarity = use_polarity
        self.height, self.width = 224, 224
        self.train = train
        self.augmentation = augmentation


        self.data_pairs = []
        for path in self.root_dir:
            self.data_pairs.extend(self._find_data_pairs(path))
    
        # Optional sanity check
        assert len(self.data_pairs) > 0, "No event/saliency pairs found."


    def _find_data_pairs(self, path):
        
        #Traverses the directory structure to find all (event_file, last_saliency_file) pairs.
        
        data_pairs = []

        n_saliency_maps = []
        N = self.event_bins
        #return many saliency maps
        for folder in os.listdir(path):
            # 1. Find the single event file (.npz)
            folder_path = os.path.join(path, folder)
            event_path = os.path.join(folder_path, self.event_folder)
            event_files = [f for f in os.listdir(event_path) if f.endswith('.npz')]

            if not event_files:
                print(f"Warning: No .npz event file found in {event_path}. Skipping.")
                continue

            event_file_path = os.path.join(event_path, event_files[0])


            # 2. Find the saliency maps (.png)
            saliency_path = os.path.join(folder_path, self.saliency_folder)
            saliency_maps = sorted([
                f for f in os.listdir(saliency_path) if f.endswith('.png')
            ])

            if not saliency_maps:
                print(f"Warning: No .png saliency map found in {saliency_path}. Skipping.")
                continue

            # 3. Find the fixation maps (.png)
            fixations_path = os.path.join(folder_path, self.fixations_folder)
            fixations_maps = sorted([
                f for f in os.listdir(fixations_path) if f.endswith('.png')
            ])

            if not fixations_maps:
                print(f"Warning: No .png fixations map found in {fixations_path}. Skipping.")
                continue

            if len(saliency_maps) < N:
                print(f" Skipping {paths}: only {len(saliency_maps)} saliency maps, need {N}")
            # We want N saliency maps
            if 'ucf' in path:
                #we shift tte start by 3 it is UCF sports dataset 
                t = 3
            else: t=0
            n_saliency_maps = [os.path.join(saliency_path, s_path) for s_path in saliency_maps[t:N+t]]
            

            n_fixation_maps = [os.path.join(fixations_path, s_path) for s_path in fixations_maps[t:N+t]]
            data_pairs.append((event_file_path, n_saliency_maps, n_fixation_maps))
            

        return data_pairs



    def load_event(self, path):
        """Loads event data from the given full path."""
        filename = os.path.basename(path)
        if filename.endswith('.npz'):
            npz = np.load(path)
            x = npz['x']
            y = npz['y']
            t = npz['t']
            p = npz['p']
            event_array = np.stack([x, y, t, p], axis=-1)  # Shape: [N, 4]
            return event_array
        else:
            raise ValueError(f"Unsupported event file format: {filename}")


    def load_saliency(self, path):
        """Loads saliency map from the given full path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Saliency map not found: {path}")

        saliency = Image.open(path).convert('L')  # grayscale
        saliency = np.array(saliency, dtype=np.float32)/ 255.0  # normalize
        saliency = cv2.resize(saliency, (self.width,self.height), interpolation=cv2.INTER_NEAREST)
        return saliency

    def load_fixation(self, path):
        """Loads fixation map from the given full path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"fixation map not found: {path}")

        fixation = Image.open(path).convert('L')  # grayscale
        fixation = np.array(fixation, dtype=np.float32)/ 255.0  # normalize
        fixation = cv2.resize(fixation, (self.width,self.height), interpolation=cv2.INTER_NEAREST)
        return fixation

    def load_n_saliency(self, paths):
        """Loads saliency map from the given full path."""

        n_sal_maps = []
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Saliency map not found: {p}")

            saliency = Image.open(p).convert('L')  # grayscale
            saliency = np.array(saliency, dtype=np.float32)/ 255.0  # normalize
            saliency = cv2.resize(saliency, (self.width,self.height), interpolation=cv2.INTER_NEAREST)

            n_sal_maps.append(saliency)
        
        n_sal_maps_tensor = np.stack(n_sal_maps)
        return n_sal_maps_tensor
    
    def load_n_fixation(self, paths):
        """Loads fixation map from the given full path."""

        n_fix_maps = []
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"fixation map not found: {p}")

            fixation = Image.open(p).convert('L')  # grayscale
            fixation = np.array(fixation, dtype=np.float32)/ 255.0  # normalize
            fixation = cv2.resize(fixation, (self.width,self.height), interpolation=cv2.INTER_NEAREST)

            n_fix_maps.append(fixation)
        
        
        n_fix_maps_tensor = np.stack(n_fix_maps)
        return n_fix_maps_tensor


    def __getitem__(self, index):
        try:
            event_path, saliency_path, fixation_path = self.data_pairs[index]
            event_data = self.load_event(event_path)
            saliency = self.load_n_saliency(saliency_path)#self.load_saliency(saliency_path)
            fixation = self.load_n_fixation(fixation_path)

        except Exception as e:
            print(f"\nException in __getitem__ (idx={index}): {repr(e)}")
            traceback.print_exc()
            raise e


        height, width = int(event_data[:, 1].max() + 1), int(event_data[:, 0].max() + 1)#self.height, self.width

        if len(event_data) < 10:
            c = 1 + int(self.event_polarity)
            event_voxel = np.zeros((self.event_bins * c, height, width), dtype=np.float32)
        else:
            if 'ucf' in str(event_path):
                event_voxel = data_util.generate_voxel_grid_V2(events=event_data,shape=(height, width),nr_temporal_bins=self.event_bins, window=100, separate_pol=self.event_polarity)#generate_6_histograms(events=event_data,shape=(height, width))#
            #event_voxel = data_util.normalize_voxel_grid_numpy(event_voxel)
            else: event_voxel = data_util.generate_voxel_grid_V2(events=event_data,shape=(height, width),nr_temporal_bins=self.event_bins, window=33.33, separate_pol=self.event_polarity)

        #saliency = cv2.resize(saliency, (event_voxel.shape[2], event_voxel.shape[1]), interpolation=cv2.INTER_NEAREST)

        
        # 2. Convert saliency to tensor and add channel dimension
        # Saliency shape: (H, W) -> (1, H, W)
        label_tensor = torch.from_numpy(saliency).unsqueeze(0).float()
        fix_tensor = torch.from_numpy(fixation).unsqueeze(0).float()
        # Convert event voxel to tensor (C, H, W)
        event_tensor = torch.from_numpy(event_voxel)

        #reshape event
        if self.event_polarity:
            # Split slices into two groups pos and neg
            pos, neg = torch.chunk(event_tensor, 2, dim=0) 
            
            event_tensor = torch.stack([pos, neg], dim=1) 
        else:
            event_tensor = event_tensor.unsqueeze(1)

        event_tensor = F.interpolate(
            event_tensor, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )

        #event_tensor = self.normalize_01(event_tensor)
        #print(event_tensor.shape)
        #print(self.data_pairs)
        return event_tensor, label_tensor, fix_tensor, event_path#.permute(1, 0, 2, 3).contiguous().squeeze(1), fix_tensor.permute(1, 0, 2, 3).contiguous().squeeze(1)
        

    def __len__(self):
        return len(self.data_pairs)
