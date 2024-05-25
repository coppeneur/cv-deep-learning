import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
import tarfile


def unpack_tar_gz(file_path):
    data_folder = 'data'
    model_folder = 'bestmodels'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created folder '{data_folder}'.")

    # Check if the target folder already contains the unpacked files
    if not os.path.exists(data_folder) or not os.listdir(data_folder):
        print(f"Unpacking {file_path} to {data_folder}...")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=data_folder)
        print("Unpacking completed.")
    else:
        print(f"Target folder '{data_folder}' is not empty. Assuming the file is already unpacked.")

    # Create the folder if it doesn't exist
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        print(f"Created folder '{model_folder}'.")

    return os.path.join(data_folder, 'fer2013', 'fer2013.csv')


class FerDataset(Dataset):

    def __init__(self, csv_file: str, transform_train=None, mode='train'):
        self.data = pd.read_csv(csv_file)

        valid_modes = {'train': 'Training', 'val': 'PublicTest', 'test': 'PrivateTest'}
        assert mode in valid_modes, f"Invalid mode: {mode}"

        self.data = self.data[self.data['Usage'] == valid_modes[mode]]
        print(f"Loaded {len(self.data)} samples for mode '{mode}'")

        if mode == 'train' and transform_train is not None:
            self.transform = transform_train
        else:
            self.transform = self.getDefaultTransform()

    def getDefaultTransform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO warum klappt der zugriff ueber den column name nicht
        # image = np.fromstring(self.data.loc[idx, 'pixels'], dtype=int, sep=' ').reshape(48, 48).astype(np.uint8)
        # label = int(self.data.loc[idx, 'emotion'])
        image = np.fromstring(self.data.iloc[idx, 1], dtype=int, sep=' ').reshape(48, 48).astype(np.uint8)
        label = int(self.data.iloc[idx, 0])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(csv_file: str, batch_size=32, transform_train=None):
    train_dataset = FerDataset(csv_file, transform_train=transform_train, mode='train')
    val_dataset = FerDataset(csv_file, mode='val')
    test_dataset = FerDataset(csv_file, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
