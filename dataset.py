import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import os
import tarfile


def unpack_tar_gz(file_path, target_folder='data'):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Check if the target folder already contains the unpacked files
    if not os.path.exists(target_folder) or not os.listdir(target_folder):
        print(f"Unpacking {file_path} to {target_folder}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=target_folder)
        print("Unpacking completed.")
    else:
        print(f"Target folder '{target_folder}' is not empty. Assuming the file is already unpacked.")


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


class FerDataset(Dataset):
    def __init__(self, csv_file: str, transform=None, mode='train'):
        self.data = pd.read_csv(csv_file)

        assert mode in ['train', 'val', 'test'], f"Invalid mode: {mode}"
        if mode == 'train':
            self.data = self.data[self.data['Usage'] == 'Training']
        elif mode == 'val':
            self.data = self.data[self.data['Usage'] == 'PublicTest']
        elif mode == 'test':
            self.data = self.data[self.data['Usage'] == 'PrivateTest']

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.fromstring(self.data['pixels'][idx], dtype=int, sep=' ').reshape(48, 48).astype(np.uint8)
        label = int(self.data['emotion'][idx])

        if self.transform:
            image = self.transform(image)

        return image, label