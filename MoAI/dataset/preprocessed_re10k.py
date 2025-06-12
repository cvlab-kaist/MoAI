import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class PreprocessedRe10k(Dataset):
    def __init__(self, data_dir, H, W, batch_size, data_dir_2=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pickle_files = [f for f in os.listdir(data_dir) if f.endswith('.pickle')]
        self.H, self.W = H, W
        
        # enable ACID sampling together
        if data_dir_2 is not None and data_dir_2 != 'None':
            self.data_dir_2 = data_dir_2
            self.pickle_files_2 = [f for f in os.listdir(data_dir_2) if f.endswith('.pickle')]
        else:
            self.data_dir_2 = None

    def __len__(self):
        return len(self.pickle_files) if self.data_dir_2 is None else len(self.pickle_files) + len(self.pickle_files_2)

    def __getitem__(self, idx):
        # dict_keys(['src', 'src_depth', 'R', 'T', 'K', 'trg', 'confs'])
        batch_src = []
        batch_src_depth = []
        batch_R = []
        batch_T = []
        batch_K = []
        batch_trg = []
        batch_confs = []

        while len(batch_src) < self.batch_size:
            
            # probability is hard-coded considering the ratio of dataset sizes.
            if random.random() < 0.2 and self.data_dir_2 is not None:
                pickle_file = random.choice(self.pickle_files_2)
            else:
                pickle_file = random.choice(self.pickle_files)
            
            
            try:
                with open(os.path.join(self.data_dir, pickle_file), 'rb') as f:
                    data = pickle.load(f)
            except pickle.UnpicklingError as e:
                continue
                            
            # with open(os.path.join(self.data_dir, pickle_file), 'rb') as f:
            #     data = pickle.load(f)

            # dict_keys(['src', 'src_depth', 'R', 'T', 'K', 'trg', 'confs'])
            src_samples = data['src']
            src_depth_samples = data['src_depth']
            R_samples = data['R']
            T_samples = data['T']
            K_samples = data['K']
            trg_samples = data['trg']
            # confs_samples = data['confs']
            
            num_samples = min(len(src_samples), self.batch_size - len(batch_src))
            indices = np.random.choice(len(src_samples), num_samples, replace=False)

            batch_src.extend(src_samples[indices])
            batch_trg.extend(trg_samples[indices])
            batch_R.extend(R_samples[indices])
            batch_T.extend(T_samples[indices])
            batch_K.extend(K_samples[indices])
            batch_src_depth.extend(src_depth_samples[indices])
            # batch_confs.extend(confs_samples[indices])

        batch_src = torch.from_numpy(np.array(batch_src)) * 2. - 1.
        batch_trg = torch.from_numpy(np.array(batch_trg)) * 2. - 1.
        batch_R = torch.from_numpy(np.array(batch_R))
        batch_T = torch.from_numpy(np.array(batch_T))
        batch_K = torch.from_numpy(np.array(batch_K))
        batch_src_depth = torch.from_numpy(np.array(batch_src_depth))
        batch_confs = torch.from_numpy(np.array(batch_confs))
        
        if self.H == batch_src.shape[2] and self.W == batch_src.shape[3]:
            pass
        elif self.H == 384 and self.W == 512:
            scaling_factor = self.H / batch_src.shape[2]
            # batch resolution 288x512 center crop to 288x384
            batch_src = batch_src[:, :, :, 64:448]
            batch_trg = batch_trg[:, :, :, 64:448]
            batch_src_depth = batch_src_depth[:, :, :, 64:448]
            # interpolate to 384x512
            batch_src = torch.nn.functional.interpolate(batch_src, size=(self.H, self.W), mode='bilinear', align_corners=False)
            batch_trg = torch.nn.functional.interpolate(batch_trg, size=(self.H, self.W), mode='bilinear', align_corners=False)
            batch_src_depth = torch.nn.functional.interpolate(batch_src_depth, size=(self.H, self.W), mode='bilinear', align_corners=False)
            # calibrate K
            batch_K[:, 0, 2] = self.W / 2
            batch_K[:, 1, 2] = self.H / 2
            batch_K[:, 0, 0] *= scaling_factor
            batch_K[:, 1, 1] *= scaling_factor
            
            

        return batch_src, batch_src_depth, batch_trg, batch_R, batch_T, batch_K