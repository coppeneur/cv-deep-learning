import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    """
    Train the given model using the given train and validation loaders, criterion, optimizer, and number of epochs.
    Will save the best model based on the validation accuracy and plot the training and validation metrics.
    :param model: model to train
    :param train_loader: training loader
    :param val_loader: validation loader
    :param criterion: criterion
    :param optimizer: optimizer
    :param num_epochs: number of epochs with default 10
    :return: None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0

    # prepare criteria and optimizer names for the pipeline title
    crit_name = str(criterion).split('(')[0].strip()
    opt_name = str(optimizer).split('(')[0].strip()

    for epoch in range(num_epochs):
        # train the model
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # validate the model
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = os.path.join("bestmodels", f"{model.get_name()}_{crit_name}_{opt_name}_best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved with accuracy: {best_val_accuracy:.4f} as '{model_path}'")

    pipeline_title = f"{model.get_name()} with {crit_name} and {opt_name} - Best Val Acc: {best_val_accuracy:.4f}"
    plot_metrics_training(train_losses, val_losses, train_accuracies, val_accuracies, pipeline_title)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the given model for one epoch using the given train loader, criterion, optimizer, and device.
    :param model: model to train
    :param train_loader: training loader
    :param criterion: criterion
    :param optimizer: optimizer
    :param device: device
    :return: average loss and accuracy for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        accuracy = 100.0 * correct / total

        progress_bar.set_postfix({'Train Loss': total_loss / (total), 'Accuracy': accuracy})

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = 100.0 * correct / total
    return avg_loss, avg_accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the given model using the given validation loader, criterion, and device.
    :param model: model to validate
    :param val_loader: validation loader
    :param criterion: criterion
    :param device: device
    :return: average loss and accuracy for the validation
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            accuracy = 100.0 * correct / total

            progress_bar.set_postfix({'Val Loss': total_loss / (total), 'Accuracy': accuracy})

    avg_loss = total_loss / len(val_loader)
    avg_accuracy = 100.0 * correct / total
    return avg_loss, avg_accuracy


def plot_metrics_training(train_losses, val_losses, train_accuracies, val_accuracies, pipeline_title: str):
    """
    Plots the training and validation losses and accuracies for the given metrics for all epochs.
    The plot is saved as a file with the pipeline title.
    :param train_losses:
    :param val_losses:
    :param train_accuracies:
    :param val_accuracies:
    :param pipeline_title:
    :return: None
    """
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

    plt.suptitle(pipeline_title, fontsize=16)
    plt.tight_layout()

    # cut off the best val accuracy from the title and save the plot
    plot_filename = os.path.join("bestmodels", f"{pipeline_title.split(' - Best Val')[0].replace(' ', '_').lower()}_plot.png")
    plt.savefig(plot_filename)

    plt.show()
