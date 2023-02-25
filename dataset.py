import numpy as np
import os
import pandas as pd
import torch
import semantic_kitti_data_loader as skdl
from torch.utils.data import Dataset

FALLING_SNOW = 110

class WADSDataset(Dataset):
    def __init__(self, res_path, config_path=None, range_max=70, intensity_max=255):
        # Set up data loader
        if(config_path == None):
            self.data_loader = skdl.SemanticKittiDataLoader(res_path)
        else:
            self.data_loader = skdl.SemanticKittiDataLoader(res_path, config_path)

        # Get target dictionary loaded
        self.dict = self.data_loader.get_label_map()
        if self.dict[FALLING_SNOW] != 'snow-falling':
            raise ValueError("label_map dictionary does not have snow-falling in correct location or does not exist.")
        
        # Set up normalization values
        self.range_max = range_max
        self.intensity_max = intensity_max
        
    def __len__(self):
        return self.data_loader.get_num_samples()
    
    def __getitem__(self, index):
        # Get targeted data and scan
        label, scan =  self.data_loader.get_data(index)

        target = np.zeros((label.shape[0], label.shape[1], 2))

        # Normalize scan to given LiDAR specs
        scan[:,:,0] = scan[:,:,0] / self.range_max
        scan[:,:,1] = scan[:,:,1] / self.intensity_max

        # Set dictionary values to 0 or 1 for non-falling snow objects and falling snow objects, respectively
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i,j] != FALLING_SNOW:
                    target[i,j,0] = 1.0
                else:
                    target[i,j,1] = 1.0

        # Move targets and scans to tensors
        scan = torch.from_numpy(scan).to(torch.float32)
        target = torch.from_numpy(target).to(torch.float32)

        # Permute the scan and targets to match standard torch format
        scan = scan.permute(2,0,1)
        target = target.permute(2,0,1)

        scan.requires_grad = True
        target.requires_grad = True

        return scan, target