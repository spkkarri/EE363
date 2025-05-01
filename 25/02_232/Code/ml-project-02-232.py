#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the relative path to the dataset
dataset_path = os.path.join(script_dir, "../data/downloaded_files/")

# In[1]:
# (No directory listing needed for local setup, keeping for reference)
# for dirname, _, filenames in os.walk(dataset_path):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# In[2]:
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn

# Define the class dictionary manually since the file is missing
class_data = {
    'name': ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown'],
    'r': [0, 255, 255, 0, 0, 255, 0],
    'g': [255, 255, 0, 255, 0, 255, 0],
    'b': [255, 0, 255, 0, 255, 255, 0]
}
class_dict = pd.DataFrame(class_data)
print("Class Dictionary:")
print(class_dict)

# Define dataset paths
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "valid")
test_dir = os.path.join(dataset_path, "test")

# Verify directories exist
for directory in [train_dir, val_dir, test_dir]:
    if os.path.exists(directory):
        print(f"{directory} exists with {len(os.listdir(directory))} files")
    else:
        print(f"Warning: {directory} does not exist")

# In[3]:
class DeepGlobeDataset(Dataset):
    def __init__(self, img_dir, is_train=True, transform=None, size=(256, 256)):
        self.img_dir = img_dir
        self.is_train = is_train  # Training mode (expect masks) or inference mode
        self.transform = transform
        self.size = size
        self.class_rgb_values = [
            [0, 255, 255],     # urban_land 
            [255, 255, 0],     # agriculture_land
            [255, 0, 255],     # rangeland
            [0, 255, 0],       # forest_land
            [0, 0, 255],       # water
            [255, 255, 255],   # barren_land
            [0, 0, 0]          # unknown
        ]
        
        # Get all satellite image paths
        self.sat_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                          if f.endswith('_sat.jpg')]
        
        # If training, filter to only include images with masks
        if self.is_train:
            valid_samples = []
            for img_path in self.sat_images:
                mask_path = img_path.replace('_sat.jpg', '_mask.png')
                if os.path.exists(mask_path):
                    valid_samples.append(img_path)
            self.sat_images = valid_samples
        
        print(f"Found {len(self.sat_images)} samples in {img_dir}")
    
    def __len__(self):
        return len(self.sat_images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.sat_images[idx]
        
        with Image.open(img_path) as image:
            image = image.resize(self.size, Image.BILINEAR)
            image = np.array(image)
        
        # Convert image to tensor and normalize
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Standard normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # For training mode, load and process mask
        if self.is_train:
            mask_path = img_path.replace('_sat.jpg', '_mask.png')
            with Image.open(mask_path) as mask:
                mask = mask.resize(self.size, Image.NEAREST)
                mask = np.array(mask)
            
            # Convert RGB mask to class indices
            mask_indices = torch.zeros(mask.shape[:2], dtype=torch.long)
            for class_idx, rgb in enumerate(self.class_rgb_values):
                r, g, b = rgb
                class_mask = (mask[:,:,0] == r) & (mask[:,:,1] == g) & (mask[:,:,2] == b)
                mask_indices[class_mask] = class_idx
            
            return image, mask_indices
        else:
            # For validation/test, return only the image
            return image, img_path

# In[4]:
# Create datasets
train_dataset = DeepGlobeDataset(train_dir, is_train=True, size=(256, 256))
val_dataset = DeepGlobeDataset(val_dir, is_train=False, size=(256, 256))
test_dataset = DeepGlobeDataset(test_dir, is_train=False, size=(256, 256))

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# In[5]:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LightAttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=7):
        super().__init__()
        
        # Reduced filter counts for memory efficiency
        # Encoder
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(img_ch, 32)  # Reduced from 64
        self.conv2 = ConvBlock(32, 64)      # Reduced from 128
        self.conv3 = ConvBlock(64, 128)     # Reduced from 256
        self.conv4 = ConvBlock(128, 256)    # Reduced from 512
        self.conv5 = ConvBlock(256, 512)    # Reduced from 1024
        
        # Attention gates with reduced dimensions
        self.attention1 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.attention2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.attention3 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.attention4 = AttentionGate(F_g=32, F_l=32, F_int=16)
        
        # Decoder
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv5 = ConvBlock(512, 256)  # 256 + 256 input channels
        
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(256, 128)  # 128 + 128 input channels
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(128, 64)   # 64 + 64 input channels
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(64, 32)    # 32 + 32 input channels
        
        # Output layer
        self.output = nn.Conv2d(32, output_ch, kernel_size=1)
        
        # Enable gradient checkpointing
        self.use_checkpointing = True

    def forward(self, x):
        # Encoding with optional checkpointing for memory efficiency
        if self.use_checkpointing and self.training:
            e1 = checkpoint.checkpoint(self.conv1, x, use_reentrant=False)
            e2 = checkpoint.checkpoint(self.conv2, self.maxpool(e1), use_reentrant=False)
            e3 = checkpoint.checkpoint(self.conv3, self.maxpool(e2), use_reentrant=False)
            e4 = checkpoint.checkpoint(self.conv4, self.maxpool(e3), use_reentrant=False)
            e5 = checkpoint.checkpoint(self.conv5, self.maxpool(e4), use_reentrant=False)
        else:
            e1 = self.conv1(x)
            e2 = self.conv2(self.maxpool(e1))
            e3 = self.conv3(self.maxpool(e2))
            e4 = self.conv4(self.maxpool(e3))
            e5 = self.conv5(self.maxpool(e4))
        
        # Decoding with attention
        d5 = self.up5(e5)
        a4 = self.attention1(g=d5, x=e4)
        d5 = torch.cat((a4, d5), dim=1)
        d5 = self.up_conv5(d5)
        
        d4 = self.up4(d5)
        a3 = self.attention2(g=d4, x=e3)
        d4 = torch.cat((a3, d4), dim=1)
        d4 = self.up_conv4(d4)
        
        d3 = self.up3(d4)
        a2 = self.attention3(g=d3, x=e2)
        d3 = torch.cat((a2, d3), dim=1)
        d3 = self.up_conv3(d3)
        
        d2 = self.up2(d3)
        a1 = self.attention4(g=d2, x=e1)
        d2 = torch.cat((a1, d2), dim=1)
        d2 = self.up_conv2(d2)
        
        output = self.output(d2)
        
        return output

# In[6]:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize your model (assuming LightAttentionUNet is defined elsewhere)
model = LightAttentionUNet().to(device)

class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.classes_to_ignore = []
        
    def forward(self, inputs, targets):
        # Remove autocast to avoid warnings
        inputs = inputs.float()
        probs = F.softmax(inputs, dim=1)
        
        dice_scores = []
        n_classes = probs.shape[1]
        
        for cls in range(n_classes):
            if cls in self.classes_to_ignore:
                continue
                
            pred_cls = probs[:, cls, ...]
            target_cls = (targets == cls).float()
            
            pred_cls = pred_cls.contiguous().view(-1)
            target_cls = target_cls.contiguous().view(-1)
            
            intersection = (pred_cls * target_cls).sum()
            cardinality = pred_cls.sum() + target_cls.sum()
            
            dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
            
            if self.weight is not None and cls < len(self.weight):
                dice = dice * self.weight[cls]
                
            dice_scores.append(dice)
        
        if len(dice_scores) > 0:
            return 1 - torch.stack(dice_scores).mean()
        else:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(weight=class_weights)
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        
        if torch.isnan(ce_loss) or torch.isnan(dice_loss):
            print(f"Warning: NaN detected in loss: CE={ce_loss}, Dice={dice_loss}")
            
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# Class weights
class_weights = torch.tensor([2.0,    # Urban
                             0.8,     # Agriculture
                             1.5,     # Rangeland
                             1.2,     # Forest
                             3.0,     # Water
                             1.7,     # Barren
                             0.5],    # Unknown/background
                            device=device)

# Define criterion and optimizer
criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0, class_weights=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=5,
    T_mult=2,
    eta_min=1e-6
)

print("Improved loss function and optimizer defined successfully")

# In[7]:
import torch
import time
from tqdm.auto import tqdm
import copy
import matplotlib.pyplot as plt

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10):
    # Track best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    history = {'train_loss': [], 'lr': []}
    
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Progress bar
        for inputs, masks in tqdm(train_loader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')
        
        # Update learning rate
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(epoch_loss)
        history['lr'].append(current_lr)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join('outputs', 'best_model.pth'))
            print(f'Model saved! Loss improved to {best_loss:.4f}')
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Load best model
    model.load_state_dict(best_model_wts)
    return model, history

# Execute training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
model, history = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

# After training, visualize results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['lr'])
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'training_history.png'))
plt.show()

# In[8]:
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the best trained model
model.load_state_dict(torch.load(os.path.join('outputs', 'best_model.pth'))['model_state_dict'])
model.eval()  # Set to evaluation mode

# Define visualization function
def visualize_prediction(model, dataset, idx, device, class_rgb_values):
    """Visualize model prediction on a single image"""
    # For validation/test dataset without ground truth
    image, img_path = dataset[idx]
    image = image.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Denormalize image for visualization
    image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = np.clip(image, 0, 1)
    
    # Create colored mask for prediction
    pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_idx, (r, g, b) in enumerate(class_rgb_values):
        pred_colored[pred == class_idx] = [r, g, b]
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Satellite Image')
    ax[0].axis('off')
    
    ax[1].imshow(pred_colored)
    ax[1].set_title('Land Cover Prediction')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', f'prediction_{os.path.basename(img_path)}.png'), dpi=300)
    
    # Calculate land cover percentages
    class_names = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]
    total_pixels = pred.size
    class_pixels = {}
    
    for class_idx, name in enumerate(class_names):
        pixels = np.sum(pred == class_idx)
        percentage = (pixels / total_pixels) * 100
        class_pixels[name] = percentage
        print(f"{name}: {percentage:.2f}%")
    
    return class_pixels

# Define class RGB values
class_rgb_values = [
    [0, 255, 255],     # urban_land 
    [255, 255, 0],     # agriculture_land
    [255, 0, 255],     # rangeland
    [0, 255, 0],       # forest_land
    [0, 0, 255],       # water
    [255, 255, 255],   # barren_land
    [0, 0, 0]          # unknown
]

# Evaluate on multiple validation images
def evaluate_model(model, val_loader, device, num_samples=5):
    """Evaluate model on validation data"""
    model.eval()
    
    # Create figure for multiple predictions
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        for i, (images, img_paths) in enumerate(val_loader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for j, (image, pred, img_path) in enumerate(zip(images, preds, img_paths)):
                if i*val_loader.batch_size + j >= num_samples:
                    break
                    
                # Denormalize image
                image = image.cpu().numpy().transpose(1, 2, 0)
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)
                
                # Create colored prediction
                pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                for class_idx, (r, g, b) in enumerate(class_rgb_values):
                    pred_colored[pred == class_idx] = [r, g, b]
                
                # Plot
                ax_idx = i*val_loader.batch_size + j
                axes[ax_idx, 0].imshow(image)
                axes[ax_idx, 0].set_title(f'Image {ax_idx+1}')
                axes[ax_idx, 0].axis('off')
                
                axes[ax_idx, 1].imshow(pred_colored)
                axes[ax_idx, 1].set_title(f'Prediction {ax_idx+1}')
                axes[ax_idx, 1].axis('off')
                
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'validation_predictions.png'), dpi=300)
    plt.show()

# Create a confusion matrix Legend showing the color for each class
def create_class_legend():
    class_names = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    
    for i, ((r, g, b), name) in enumerate(zip(class_rgb_values, class_names)):
        color = [r/255, g/255, b/255]
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.9, color=color))
        ax.text(i+0.45, 0.5, name, ha='center', va='center', 
                fontsize=9, fontweight='bold', 
                color='white' if sum(color) < 1.5 else 'black')
    
    plt.xlim(0, len(class_names))
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join('outputs', 'class_legend.png'), dpi=300)
    plt.show()

# Execute evaluation and visualization
print("Starting model evaluation and visualization...")
if isinstance(val_dataset[0][1], str):  # Check if val_dataset returns image paths
    # For datasets without ground truth
    for i in range(5):  # Visualize 5 random images
        idx = np.random.randint(0, len(val_dataset))
        print(f"\nVisualization {i+1}:")
        visualize_prediction(model, val_dataset, idx, device, class_rgb_values)
else:
    # For datasets with ground truth
    evaluate_model(model, val_loader, device)

# Create legend for classes
create_class_legend()

# Optional: Save the model in a deployable format (ONNX)
dummy_input = torch.randn(1, 3, 256, 256, device=device)
torch.onnx.export(model, dummy_input, os.path.join('outputs', 'vegetation_segmentation_model.onnx'), 
                 verbose=True, input_names=['input'], output_names=['output'])
print("Model exported to ONNX format for deployment")