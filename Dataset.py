import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader

# %%
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


# Define the file path and target folder
file_path = 'fer2013.tar.gz'

unpack_tar_gz(file_path, target_folder)
#%%
# dataset is available at https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge
df = pd.read_csv('data/fer2013/fer2013.csv')
print(df.shape)
df.head()
#%%
df['Usage'].value_counts()
#%%
df['emotion'].value_counts()
#%%
# plot the distribution of the emotions
df['emotion'].value_counts().plot(kind='bar')