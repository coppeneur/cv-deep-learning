import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import numpy as np


def evaluate_model(model, test_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}%')

    plot_confusion_matrix(all_labels, all_preds, class_names)
    print_classification_report(all_labels, all_preds, class_names)


def plot_confusion_matrix(labels, preds, classes, normalize=True):
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
    plt.show()


def print_classification_report(labels, preds, classes):
    report = classification_report(labels, preds, target_names=classes, zero_division=0)
    print('Classification Report:\n')
    print(report)


def plot_metrics_training(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_activations(model, input_image):
    model.eval()

    activations = {}

    def hook(module, input, output):
        activations[module] = output.detach()

    hooks = [module.register_forward_hook(hook)
             for name, module in model.named_modules()
             if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]

    with torch.no_grad():
        _ = model(input_image)

    for hook_handle in hooks:
        hook_handle.remove()

    return activations


def plot_activations(model, input_image, emotion_name):
    if isinstance(input_image, torch.Tensor) and input_image.dim() == 3:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        input_image = transform(input_image).unsqueeze(0)

    activations = get_activations(model, input_image)

    for layer, activation in activations.items():
        if len(activation.shape) == 4:
            num_channels = activation.shape[1]
            num_rows = (num_channels + 3) // 4
            plt.figure(figsize=(12, 3 * num_rows))
            plt.suptitle(emotion_name, fontsize=16, fontweight='bold', color='black')
            for i in range(num_channels):
                plt.subplot(num_rows, 4, i+1)
                plt.imshow(activation[0, i].cpu(), cmap='gray')
                plt.title(f'Channel: {i+1}', fontsize=10, fontweight='bold', color='black')
                plt.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        break  # Only plot the activations for the first layer