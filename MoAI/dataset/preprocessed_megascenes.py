import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class PreprocessedMegaScenes(Dataset):
    def __init__(self, data_dir, H, W, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.src_files = [f for f in os.listdir(os.path.join(data_dir,'src')) if f.endswith('.npy')]
        self.H, self.W = H, W

        assert H==384 and W==512, 'Only support 384x512 resolution'

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        batch_src = []
        batch_src_depth = []
        batch_R = []
        batch_T = []
        batch_K = []
        batch_trg = []
        batch_confs = []

        while len(batch_src) < self.batch_size:
            
            src_files = random.choice(self.src_files)
            
            try:
                # K  R  T  confs  src  src_depth  trg
                src_npy = np.load(os.path.join(self.data_dir,'src', src_files))
                src_depth_npy = np.load(os.path.join(self.data_dir,'src_depth', src_files))
                R_npy = np.load(os.path.join(self.data_dir,'R', src_files))
                T_npy = np.load(os.path.join(self.data_dir,'T', src_files))
                K_npy = np.load(os.path.join(self.data_dir,'K', src_files))
                trg_npy = np.load(os.path.join(self.data_dir,'trg', src_files))
                confs_npy = np.load(os.path.join(self.data_dir,'confs', src_files))
            except Exception as e:
                continue

            num_samples = min(len(src_npy), self.batch_size - len(batch_src))
            indices = np.random.choice(len(src_npy), num_samples, replace=False)

            batch_src.extend(src_npy[indices])
            batch_trg.extend(trg_npy[indices])
            batch_R.extend(R_npy[indices])
            batch_T.extend(T_npy[indices])
            batch_K.extend(K_npy[indices])
            batch_src_depth.extend(src_depth_npy[indices])
            batch_confs.extend(confs_npy[indices])

        batch_src = torch.from_numpy(np.array(batch_src)) * 2. - 1.
        batch_trg = torch.from_numpy(np.array(batch_trg)) * 2. - 1.
        batch_R = torch.from_numpy(np.array(batch_R))
        batch_T = torch.from_numpy(np.array(batch_T))
        batch_K = torch.from_numpy(np.array(batch_K))
        batch_src_depth = torch.from_numpy(np.array(batch_src_depth))
        batch_confs = torch.from_numpy(np.array(batch_confs))

        return batch_src, batch_src_depth, batch_trg, batch_R, batch_T, batch_K