import torch
import random
import h5py
import numpy as np
import os

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.98, patience=10, verbose=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.9)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.991)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 20, 30], gamma=0.1)
    
def set_seed(seed: int = 0):    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
