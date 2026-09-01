import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np




class GaussianBlur3D(nn.Module):
    def __init__(self, sigma=3, kernel_size=25):
        super().__init__()
        self.register_buffer("kernel", self.gaussian_kernel_3d(1, kernel_size, sigma))

    def gaussian_kernel_3d(self, channels=1, kernel_size=25, sigma=3):
        # coordinate grid
        ax = torch.arange(kernel_size) - kernel_size // 2
        tt, yy, xx = torch.meshgrid(ax, ax, ax, indexing='ij')  
        # shapes: (kernel_size, kernel_size, kernel_size)

        # 3D Gaussian formula
        kernel = torch.exp(-(tt**2 + yy**2 + xx**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # conv3d: (out_channels, in_channels/groups, kT, kH, kW)
        kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        return kernel

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        #padding in all dimensions
        pad = self.kernel.size(-1) // 2  

        # Expand kernel to match number of channels (grouped convolution)
        k = self.kernel.expand(C, 1, -1, -1, -1).to(x.device)

        return F.conv3d(x, k, padding=pad, groups=C)



class Bias(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, tensor):
        # For 5D input: (N, C, D, H, W)
        return tensor + self.bias[None, :, None, None, None]



class CenterBias(nn.Module):
    def __init__(self, H=224, W=224):
        super().__init__()
        # Random positive bias between 0 and 1
        init_map = torch.rand(H, W)
        init_map = init_map / init_map.max()  
        # Register as parameter with shape (1, 1, H, W)
        self.bias = nn.Parameter(init_map[None, None, :, :])

    def forward(self, x):
        # x: (B, T, 1, H, W)
        return x * (1 +self.bias) 


class cnn_decoder(nn.Module):
    def __init__(self, in_channels_list, num_classes=1, fpn_channels=256, debug=True):
        super().__init__()
        
        # Lateral Convs: Reduce channels of all feature maps
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(c, fpn_channels, kernel_size=1)
            for c in in_channels_list
        ])
        
        # Total number of channels after upsampling and concatenating all features
        total_fused_channels = len(in_channels_list) * fpn_channels
        
        self.final_projection = nn.Sequential(
            nn.Conv3d(total_fused_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fpn_channels),
            Bias(fpn_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # 3. Final 1x1 Conv to output the desired number of classes
        self.final_pred = nn.Conv3d(fpn_channels, num_classes, kernel_size=3, padding=1) #nn.Conv3d(fpn_channels, num_classes, kernel_size=(1,3,3), padding=(0, 1, 1))
        nn.init.zeros_(self.final_pred.bias)


    def forward(self, features, input_size=None):
        if input_size is None:
            if features:
                target_size = features[0].shape[2:]
            else:
                raise ValueError("Features list is empty.")
        else:
            target_size = input_size
        
        upsampled_features = []
        
        for i, feature in enumerate(features):
            # Reduce channels
            lat = self.lateral_convs[i](feature)
            
            # Upsample 
            up = F.interpolate(lat, size=target_size, mode="trilinear", align_corners=False)
            upsampled_features.append(up)
            
        # Concatenate all upsampled features along the channel dimension
        fused = torch.cat(upsampled_features, dim=1) 
        
        
        # predict
        projected = self.final_projection(fused)
        sal = self.final_pred(projected)
        
        return sal




class cnn_decoder_ablation(nn.Module):
    def __init__(self, in_channels_list, num_classes=1, fpn_channels=256, debug=True):
        super().__init__()
        
        # Lateral Convs: Reduce channels of all feature maps
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(c, fpn_channels, kernel_size=1)
            for c in in_channels_list
        ])
        
        # Total number of channels after upsampling and concatenating all features
        total_fused_channels = len(in_channels_list) * fpn_channels
        
        self.final_projection = nn.Sequential(
            nn.Conv3d(total_fused_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fpn_channels),
            Bias(fpn_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # 3. Final 1x1 Conv to output the desired number of classes
        self.final_pred = nn.Conv3d(fpn_channels, num_classes, kernel_size=1, padding=(0, 1, 1)) #nn.Conv3d(fpn_channels, num_classes, kernel_size=(1,3,3), padding=(0, 1, 1))
        nn.init.zeros_(self.final_pred.bias)


    def forward(self, features, input_size=None):
        if input_size is None:
            if features:
                target_size = features[0].shape[2:]
            else:
                raise ValueError("Features list is empty.")
        else:
            target_size = input_size
        
        upsampled_features = []
        
        for i, feature in enumerate(features):
            # Reduce channels
            lat = self.lateral_convs[i](feature)
            
            # Upsample 
            up = F.interpolate(lat, size=target_size, mode="trilinear", align_corners=False)
            upsampled_features.append(up)
            
        # Concatenate all upsampled features along the channel dimension
        fused = torch.cat(upsampled_features, dim=1) 
        
        
        # predict
        projected = self.final_projection(fused)
        sal = self.final_pred(projected)
        
        return sal



class SaliencyEncoderDecoder(nn.Module):
    
    def __init__(self,
                 backbone,
                 opt, 
                 ): 
        super().__init__()
        
        self.backbone = backbone
        self.opt = opt
        
        #gaussian blur
        self.gaussian_blur = GaussianBlur3D()
        #center bias
        self.center_bias = CenterBias()


        self.align_corners = False 

        print(opt['network'])
        if self.opt['network']["decoder"] == 'cnn':
            self._init_decode_head(self.opt["head_main"])

        
    def _init_decode_head(self, decode_head_config):
        # parameters from your YAML 'head_main'
        in_channels_list = decode_head_config['in_channels']
        num_classes = decode_head_config['num_classes'] 
        
        # SaliencyHead 
        self.decode_head = cnn_decoder(
            in_channels_list=in_channels_list,
            num_classes=num_classes
        )
        self.num_classes = num_classes

    def extract_feat(self, img):
        B, T, C, H, W = img.shape
        x, x_downsample = self.backbone(img, B, T, W)
        return x, x_downsample, T

    def forward(self, img):
        B, T, C, H, W = img.shape

        #Encoder features extraction
        x_feats, x_downsample, T= self.extract_feat(img) 
        
        
        #Decoding
        if self.opt['network']["decoder"] == 'cnn':
           saliency_map = self.decode_head(x_downsample)

           saliency_map = saliency_map.permute(0, 2, 1, 3, 4).contiguous()
           saliency_map = F.interpolate(
                input=saliency_map,
                size=(1,H,W), # Target H, W from input image
                mode='trilinear',
                align_corners=self.align_corners
            )
        else: 
            saliency_map = self.swin_decoder(x_feats, x_downsample, T)


        #center bias + Gaussian blur
        saliency_map = self.center_bias(saliency_map)       
        saliency_map = self.gaussian_blur(saliency_map) 

        return saliency_map

