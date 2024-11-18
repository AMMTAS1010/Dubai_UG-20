import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os
import time
import copy
import matplotlib.pyplot as plt

# CNN Classifier Functions

def load_classes(classes_file):
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    return class_to_idx

class Food101Dataset(Dataset):
    def __init__(self, file_list, root_dir, transform=None):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        relative_path, label = self.file_list[idx]
        full_path = os.path.join(self.root_dir, relative_path) + ".jpg"
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File {full_path} not found.")
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def load_file_list(list_file, root_dir, class_to_idx):
    file_list = []
    with open(list_file, 'r') as f:
        for line in f:
            relative_path = line.strip()
            label = relative_path.split('/')[0]
            label_index = class_to_idx[label]
            file_list.append((relative_path, label_index))
    return file_list

def train_cnn_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs, results_dir):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # Convert running_corrects to float32 instead of float64
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            epoch_acc_value = epoch_acc.item()  # Convert tensor to Python float

            if phase == 'train':
                train_acc_history.append(epoch_acc_value)
            else:
                val_acc_history.append(epoch_acc_value)

            print(f'{phase} Acc: {epoch_acc_value:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('-' * 10)

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Acc: {best_acc.item():.4f}')

    # Plot training and validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_acc_history, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_acc_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir, 'cnn_training_accuracy.png'))
    plt.show()

    # Save the trained model
    torch.save(best_model_wts, os.path.join(results_dir, 'best_model.pth'))

    model.load_state_dict(best_model_wts)
    return model

def create_cnn_model(dataset_path, meta_path, num_classes, results_dir, img_height=224, img_width=224, batch_size=64, epochs=10):
    # Device selection optimized for M1 Mac GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    train_list_file = os.path.join(meta_path, 'train.txt')
    test_list_file = os.path.join(meta_path, 'test.txt')
    classes_file = os.path.join(meta_path, 'classes.txt')
    class_to_idx = load_classes(classes_file)

    train_file_list = load_file_list(train_list_file, dataset_path, class_to_idx)
    test_file_list = load_file_list(test_list_file, dataset_path, class_to_idx)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop((img_height, img_width), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    train_dataset = Food101Dataset(train_file_list, dataset_path, transform=data_transforms['train'])
    test_dataset = Food101Dataset(test_file_list, dataset_path, transform=data_transforms['val'])
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
    }
    dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    model = train_cnn_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, epochs, results_dir)
    return model