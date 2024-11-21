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
    """
    Loads class names from a TXT file and creates a mapping from class name to index.
    
    Args:
        classes_file (str): Path to the classes.txt file.
    
    Returns:
        dict: Mapping from class name to class index.
    """
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip() != '']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    print(f"Loaded {len(class_to_idx)} classes.")
    return class_to_idx

def load_file_list(list_file, root_dir, class_to_idx):
    """
    Loads image paths and corresponding labels from a TXT file.
    
    Args:
        list_file (str): Path to the list file (train.txt or test.txt).
        root_dir (str): Root directory containing all images.
        class_to_idx (dict): Mapping from class name to class index.
    
    Returns:
        list: List of tuples containing (relative_path, label_index).
    """
    file_list = []
    with open(list_file, 'r') as f:
        for line in f:
            relative_path = line.strip()
            if relative_path == '':
                continue  # Skip empty lines
            label = relative_path.split('/')[0]
            if label not in class_to_idx:
                print(f"Warning: Label '{label}' not found in class_to_idx mapping. Skipping '{relative_path}'.")
                continue
            label_index = class_to_idx[label]
            file_list.append((relative_path, label_index))
    print(f"Loaded {len(file_list)} samples from '{list_file}'.")
    return file_list

class Food101Dataset(Dataset):
    """
    Custom Dataset for loading Food101 images.
    Assumes that the file_list contains tuples of (relative_path, label_index).
    Images are expected to have a '.jpg' extension.
    """
    def __init__(self, file_list, root_dir, transform=None):
        """
        Initializes the dataset with a list of image paths and labels.
        
        Args:
            file_list (list): List of tuples containing (relative_path, label_index).
            root_dir (str): Root directory containing all images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.file_list = file_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        relative_path, label = self.file_list[idx]
        full_path = os.path.join(self.root_dir, relative_path) + ".jpg"
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File '{full_path}' not found.")
        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image '{full_path}': {e}")
            # Return a black image if corrupted
            image = Image.new('RGB', (self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label

class ResidualBlock(nn.Module):
    """
    A residual block for the EnhancedClassicalCNN.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initializes the ResidualBlock.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride for the convolution. Defaults to 1.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        """
        Forward pass of the ResidualBlock.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying residual connection.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class EnhancedClassicalCNN(nn.Module):
    """
    An enhanced classical CNN architecture for image classification with residual connections and global average pooling.
    """
    def __init__(self, num_classes, input_size=(224, 224)):
        """
        Initializes the EnhancedClassicalCNN model.
        
        Args:
            num_classes (int): Number of output classes.
            input_size (tuple, optional): Size of the input images. Defaults to (224, 224).
        """
        super(EnhancedClassicalCNN, self).__init__()
        self.features = nn.Sequential(
            # Initial Convolution
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual Blocks
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output logits.
        """
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten starting from dimension 1
        x = self.classifier(x)
        return x

def train_cnn_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs, results_dir, patience=5):
    """
    Trains the CNN model with early stopping and tracks the best model based on validation accuracy.
    
    Args:
        model (nn.Module): The CNN model to train.
        criterion (loss function): Loss function.
        optimizer (Optimizer): Optimizer.
        scheduler (Scheduler): Learning rate scheduler.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        dataset_sizes (dict): Dictionary containing sizes of 'train' and 'val' datasets.
        device (torch.device): Device to train on.
        num_epochs (int): Maximum number of training epochs.
        results_dir (str): Directory to save results (plots, models).
        patience (int): Number of epochs to wait for improvement before stopping.
    
    Returns:
        nn.Module: The best trained model based on validation accuracy.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    train_acc_history = []
    val_acc_history = []

    # To keep track of all checkpoints
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            epoch_acc_value = epoch_acc.item()

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc_value:.4f}')

            # Record accuracy history
            if phase == 'train':
                train_acc_history.append(epoch_acc_value)
            else:
                val_acc_history.append(epoch_acc_value)
                # Check for improvement
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    # Save checkpoint
                    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_acc_{best_acc:.4f}.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f'Checkpoint saved at {checkpoint_path}')
                else:
                    epochs_no_improve += 1

        # Scheduler step
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            # Assuming 'val' is the last phase and has the latest epoch_acc_value
            scheduler.step(epoch_acc)
            print(f"Scheduler step with validation accuracy: {epoch_acc_value:.4f}")
        else:
            scheduler.step()
            print("Scheduler step called.")

        print()  # Newline for better readability

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs with no improvement.')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation Accuracy: {best_acc.item():.4f}')

    # Plot training and validation accuracy
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(train_acc_history)+1), train_acc_history, label='Training Accuracy')
    plt.plot(range(1, len(val_acc_history)+1), val_acc_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'cnn_training_accuracy.png'))
    plt.show()

    # Save the best model weights
    torch.save(best_model_wts, os.path.join(results_dir, 'best_model.pth'))
    print(f'Best model saved at {os.path.join(results_dir, "best_model.pth")}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def create_cnn_model(dataset_path, meta_path, num_classes, results_dir, img_height=224, img_width=224, batch_size=64, epochs=50, use_pretrained=True):
    """
    Creates and trains a CNN model based on the specified parameters.
    
    Args:
        dataset_path (str): Path to the dataset images.
        meta_path (str): Path to the metadata files (classes.txt, train.txt, test.txt).
        num_classes (int): Number of classes in the dataset.
        results_dir (str): Directory to save results (plots, models).
        img_height (int, optional): Height of input images. Defaults to 224.
        img_width (int, optional): Width of input images. Defaults to 224.
        batch_size (int, optional): Batch size for training. Defaults to 64.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        use_pretrained (bool, optional): If True, use a pre-trained ResNet18. If False, use an enhanced classical CNN. Defaults to True.
    
    Returns:
        nn.Module: The best trained model based on validation accuracy.
    """
    
    # Device selection optimized for M1 Mac GPU or CUDA
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load class-to-index mapping
    classes_file = os.path.join(meta_path, 'classes.txt')
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"Required file '{classes_file}' not found in meta_path '{meta_path}'.")
    class_to_idx = load_classes(classes_file)

    # Load file lists
    train_list_file = os.path.join(meta_path, 'train.txt')
    test_list_file = os.path.join(meta_path, 'test.txt')

    if not os.path.exists(train_list_file):
        raise FileNotFoundError(f"Required file '{train_list_file}' not found in meta_path '{meta_path}'.")
    if not os.path.exists(test_list_file):
        raise FileNotFoundError(f"Required file '{test_list_file}' not found in meta_path '{meta_path}'.")

    train_file_list = load_file_list(train_list_file, dataset_path, class_to_idx)
    test_file_list = load_file_list(test_list_file, dataset_path, class_to_idx)

    print(f'Total Training Samples: {len(train_file_list)}')
    print(f'Total Validation Samples: {len(test_file_list)}')

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_height, img_width)),  # Resizing to fixed size
            transforms.RandomResizedCrop((img_height, img_width), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_height, img_width)),  # Resizing to fixed size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    # Create datasets
    train_dataset = Food101Dataset(train_file_list, dataset_path, transform=data_transforms['train'])
    test_dataset = Food101Dataset(test_file_list, dataset_path, transform=data_transforms['val'])

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True),
        'val': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True),
    }

    # Get dataset sizes
    dataset_sizes = {'train': len(train_dataset), 'val': len(test_dataset)}

    # Initialize the model
    if use_pretrained:
        print("Initializing ResNet18 with pre-trained weights.")
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the final fully connected layer
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
    else:
        print("Initializing an enhanced classical CNN from scratch.")
        model = EnhancedClassicalCNN(num_classes=num_classes, input_size=(img_height, img_width))
    
    model = model.to(device)

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    
    if use_pretrained:
        # Train all layers with a smaller learning rate
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)
    else:
        # Use SGD with momentum and weight decay
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Train the model with Early Stopping
    model = train_cnn_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, epochs, results_dir, patience=7)
    return model