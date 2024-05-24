import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

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

    return os.path.join(target_folder, 'fer2013', 'fer2013.csv')

class FerDataset(Dataset):
    def __init__(self, csv_file: str, transform=None, mode='train'):
        self.data = pd.read_csv(csv_file)

        assert mode in ['train', 'val', 'test'], f"Invalid mode: {mode}"
        if mode == 'train':
            self.data = self.data[self.data['Usage'] == 'Training']
            description = self.data.describe()
            print(description)
        elif mode == 'val':
            print("val mode activated")
            self.data = self.data[self.data['Usage'] == 'PublicTest']
            print(len(self.data))
            description = self.data.describe()
            print(description)
        elif mode == 'test':
            self.data = self.data[self.data['Usage'] == 'PrivateTest']

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO warum klappt der zugriff ueber den column name nicht
        #image = np.fromstring(self.data.loc[idx, 'pixels'], dtype=int, sep=' ').reshape(48, 48).astype(np.uint8)
        #label = int(self.data.loc[idx, 'emotion'])
        image = np.fromstring(self.data.iloc[idx, 1], dtype=int, sep=' ').reshape(48, 48).astype(np.uint8)
        label = int(self.data.iloc[idx, 0])


        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(file_path: str, train_transform: transforms.Compose, test_val_transform: transforms.Compose,
                    batch_size: int):
    train_dataset = FerDataset(file_path, transform=train_transform, mode='train')
    val_dataset = FerDataset(file_path, transform=test_val_transform, mode='val')
    test_dataset = FerDataset(file_path, transform=test_val_transform, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
