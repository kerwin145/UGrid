import os
import typing

from torch.utils.data import Dataset
import numpy as np
import torch
import random


class SynDat(Dataset):
    def __init__(self, dataset_root: str, use_data: float = 1.0):
        super().__init__()
        self.dataset_root: str = dataset_root
        self.instance_path_lst: typing.List[str] = []

        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in sorted(filenames):
                self.instance_path_lst.append(os.path.join(dirpath, filename))

        if use_data < 1.0:
            random.seed(123)
            random.shuffle(self.instance_path_lst)
            keep = int(len(self.instance_path_lst) * use_data)
            # print(keep)
            # print(len(self.instance_path_lst))
            self.instance_path_lst = self.instance_path_lst[:keep]
        
        # print(f"HEHEH(HF(WHE(FHW ))) {len(self.instance_path_lst)}")

    def __getitem__(self, index: int):
        data: np.ndarray = np.load(self.instance_path_lst[index])
        bc_value, bc_mask = data
        bc_value: torch.Tensor = torch.from_numpy(bc_value).float().unsqueeze(0)
        bc_mask: torch.Tensor = torch.from_numpy(bc_mask).float().unsqueeze(0)
        return {'bc_value': bc_value,
                'bc_mask': bc_mask}

    def __len__(self):
        return len(self.instance_path_lst) // 4
