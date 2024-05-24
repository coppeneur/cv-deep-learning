# this file contains the training function for a given CNN model.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    n_train_steps = len(train_loader)
    n_val_steps = len(val_loader)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, n_train_steps)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device, n_val_steps)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    plot_losses(train_losses, val_losses)

    return train_losses, val_losses


def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train_one_epoch(model, train_loader, criterion, optimizer, device, n_steps):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(enumerate(train_loader), total=n_steps, desc="Training")
    for i, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Train Loss': total_loss / (i + 1)})

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, n_steps):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(enumerate(val_loader), total=n_steps, desc="Validation")
    with torch.no_grad():
        for i, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            progress_bar.set_postfix({'Val Loss': total_loss / (i + 1)})

    return total_loss / len(val_loader)
