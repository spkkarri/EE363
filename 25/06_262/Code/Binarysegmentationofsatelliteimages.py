
# %%
# Mount Google Drive


#from google.colab import drive
#drive.mount('/content/drive')

# Install necessary libraries
#!pip install timm albumentations segmentation-models-pytorch --quiet



# %%
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
# %%
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm
# %%
from torch.utils.data import DataLoader



# %%
class DeepGlobeDataset(Dataset):


    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.image_paths)

# List paths
image_dir = '/content/drive/MyDrive/archive (2)/data/data/training_data/images'
mask_dir = '/content/drive/MyDrive/archive (2)/data/data/training_data/masks'

image_paths = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir)])
mask_paths = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir)])

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

dataset = DeepGlobeDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# %%
import segmentation_models_pytorch as smp

class BinarySegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)

# %%
'''import os
import numpy as np
from PIL import Image'''

# RGB to class index map
color_to_class = {
    (0, 255, 255): 0,    # Urban
    (255, 255, 0): 1,    # Agriculture
    (255, 0, 255): 2,    # Rangeland
    (0, 255, 0): 3,      # Forest
    (0, 0, 255): 4,      # Water
    (255, 255, 255): 5,  # Barren
    (0, 0, 0): 6         # Unknown
}

# Input/output paths
multi_class_mask_dir = '/content/drive/MyDrive/archive (2)/data/data/training_data/masks'
output_dir = '/content/drive/MyDrive/archive (2)/data/data/training_data/binarymasks_rgb'
os.makedirs(output_dir, exist_ok=True)

# Process each image
for fname in os.listdir(multi_class_mask_dir):
    if not fname.endswith('.png'):
        continue

    mask = np.array(Image.open(os.path.join(multi_class_mask_dir, fname)))  # shape: (H, W, 3)

    for rgb, class_idx in color_to_class.items():
        binary_mask = np.all(mask == rgb, axis=-1).astype(np.uint8) * 255

        out_path = os.path.join(output_dir, f"{fname.replace('.png', '')}_class_{class_idx}.png")
        Image.fromarray(binary_mask).save(out_path)


# %%
''''import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm'''

# === Dataset Class ===
class BinarySegDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)  # shape: (1, H, W)

        return image, mask

# === Transform ===
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# === Paths ===
image_dir = "/content/drive/MyDrive/archive (2)/data/data/training_data/images"
binary_mask_dir = "/content/drive/MyDrive/archive (2)/data/data/training_data/binarymasks_rgb"

# === Training Loop for Each Class ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
num_classes = 7

for class_id in range(num_classes):
    print(f"\n🔁 Training binary segmentation model for Class {class_id}\n")

    # === Collect image and class-specific mask paths ===
    image_paths, mask_paths = [], []
    for fname in os.listdir(image_dir):
        if fname.endswith("_sat.jpg"):
            image_id = fname.replace("_sat.jpg", "")
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(binary_mask_dir, f"{image_id}_mask_class_{class_id}.png")

            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)

    if len(image_paths) == 0:
        print(f"⚠️ No masks found for class {class_id}. Skipping...")
        continue

    # === Split ===
    train_img = image_paths[:int(0.8 * len(image_paths))]
    train_msk = mask_paths[:int(0.8 * len(mask_paths))]
    val_img = image_paths[int(0.8 * len(image_paths)):]
    val_msk = mask_paths[int(0.8 * len(mask_paths)):]

    # === Dataset & Dataloader ===
    train_ds = BinarySegDataset(train_img, train_msk, transform)
    val_ds = BinarySegDataset(val_img, val_msk, transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    # === Model ===
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # === Training Loop ===
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"[Class {class_id}] Epoch [{epoch+1}/{EPOCHS}]")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False).squeeze(1)
            loss = criterion(outputs, masks.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"✅ [Class {class_id}] Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    # === Save Model ===
    save_dir = "/content/drive/MyDrive/binary_seg_models"  # Folder in your Drive
    os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

    model_path = os.path.join(save_dir, f"binary_segformer_class_{class_id}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model for class {class_id} at {model_path}")

# %%
'''import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm'''

# === Dice and IoU functions ===
def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice.item()

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.item()

# === Constants ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "/content/drive/MyDrive/binary_seg_models"
num_classes = 7

# === Store results ===
results = []

# === Loop over all 7 classes ===
for class_id in range(num_classes):
    print(f"\n📊 Evaluating Model for Class {class_id}")

    # === Load validation dataset for current class ===
    val_img = []
    val_msk = []

    for fname in os.listdir(image_dir):
        if fname.endswith("_sat.jpg"):
            image_id = fname.replace("_sat.jpg", "")
            mask_path = os.path.join(binary_mask_dir, f"{image_id}_mask_class_{class_id}.png")
            if os.path.exists(mask_path):
                val_img.append(os.path.join(image_dir, fname))
                val_msk.append(mask_path)

    val_img = val_img[int(0.8 * len(val_img)):]
    val_msk = val_msk[int(0.8 * len(val_msk)):]

    val_ds = BinarySegDataset(val_img, val_msk, transform)
    val_loader = DataLoader(val_ds, batch_size=4)

    # === Load model ===
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    model_path = os.path.join(save_dir, f"binary_segformer_class_{class_id}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # === Evaluate ===
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            outputs = torch.sigmoid(outputs)

            for pred, gt in zip(outputs, masks):
                dice_scores.append(dice_score(pred, gt))
                iou_scores.append(iou_score(pred, gt))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print(f"✅ Class {class_id}: Dice = {avg_dice:.4f}, IoU = {avg_iou:.4f}")
    results.append((class_id, avg_dice, avg_iou))

# === Optional: Save results as CSV ===
import pandas as pd

results_df = pd.DataFrame(results, columns=["Class", "Dice Score", "IoU"])
results_df.to_csv("/content/drive/MyDrive/binary_seg_eval_results.csv", index=False)




# %%
%env CUDA_LAUNCH_BLOCKING=1

# %%
# Mount Google Drive


#from google.colab import drive
#drive.mount('/content/drive') #this does not wokr in local folder

# Install necessary libraries
#!pip install timm albumentations segmentation-models-pytorch --quiet



# %%
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
# %%
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm
# %%
from torch.utils.data import DataLoader



# %%
class DeepGlobeDataset(Dataset):


    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.image_paths)

# List paths
image_dir = '/content/drive/MyDrive/archive (2)/data/data/training_data/images'
mask_dir = '/content/drive/MyDrive/archive (2)/data/data/training_data/masks'

image_paths = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir)])
mask_paths = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir)])

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

dataset = DeepGlobeDataset(image_paths, mask_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# %%
import segmentation_models_pytorch as smp

class BinarySegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)

# %%
'''import os
import numpy as np
from PIL import Image'''

# RGB to class index map
color_to_class = {
    (0, 255, 255): 0,    # Urban
    (255, 255, 0): 1,    # Agriculture
    (255, 0, 255): 2,    # Rangeland
    (0, 255, 0): 3,      # Forest
    (0, 0, 255): 4,      # Water
    (255, 255, 255): 5,  # Barren
    (0, 0, 0): 6         # Unknown
}

# Input/output paths
multi_class_mask_dir = '/content/drive/MyDrive/archive (2)/data/data/training_data/masks'
output_dir = '/content/drive/MyDrive/archive (2)/data/data/training_data/binarymasks_rgb'
os.makedirs(output_dir, exist_ok=True)

# Process each image
for fname in os.listdir(multi_class_mask_dir):
    if not fname.endswith('.png'):
        continue

    mask = np.array(Image.open(os.path.join(multi_class_mask_dir, fname)))  # shape: (H, W, 3)

    for rgb, class_idx in color_to_class.items():
        binary_mask = np.all(mask == rgb, axis=-1).astype(np.uint8) * 255

        out_path = os.path.join(output_dir, f"{fname.replace('.png', '')}_class_{class_idx}.png")
        Image.fromarray(binary_mask).save(out_path)


# %%
''''import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm'''

# === Dataset Class ===
class BinarySegDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)  # shape: (1, H, W)

        return image, mask

# === Transform ===
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# === Paths ===
image_dir = "/content/drive/MyDrive/archive (2)/data/data/training_data/images"
binary_mask_dir = "/content/drive/MyDrive/archive (2)/data/data/training_data/binarymasks_rgb"

# === Training Loop for Each Class ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
num_classes = 7

for class_id in range(num_classes):
    print(f"\n🔁 Training binary segmentation model for Class {class_id}\n")

    # === Collect image and class-specific mask paths ===
    image_paths, mask_paths = [], []
    for fname in os.listdir(image_dir):
        if fname.endswith("_sat.jpg"):
            image_id = fname.replace("_sat.jpg", "")
            img_path = os.path.join(image_dir, fname)
            mask_path = os.path.join(binary_mask_dir, f"{image_id}_mask_class_{class_id}.png")

            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)

    if len(image_paths) == 0:
        print(f"⚠️ No masks found for class {class_id}. Skipping...")
        continue

    # === Split ===
    train_img = image_paths[:int(0.8 * len(image_paths))]
    train_msk = mask_paths[:int(0.8 * len(mask_paths))]
    val_img = image_paths[int(0.8 * len(image_paths)):]
    val_msk = mask_paths[int(0.8 * len(mask_paths)):]

    # === Dataset & Dataloader ===
    train_ds = BinarySegDataset(train_img, train_msk, transform)
    val_ds = BinarySegDataset(val_img, val_msk, transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    # === Model ===
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # === Training Loop ===
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"[Class {class_id}] Epoch [{epoch+1}/{EPOCHS}]")

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False).squeeze(1)
            loss = criterion(outputs, masks.squeeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"✅ [Class {class_id}] Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    # === Save Model ===
    save_dir = "/content/drive/MyDrive/binary_seg_models"  # Folder in your Drive
    os.makedirs(save_dir, exist_ok=True)  # Create the folder if it doesn't exist

    model_path = os.path.join(save_dir, f"binary_segformer_class_{class_id}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model for class {class_id} at {model_path}")

# %%
'''import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm'''

# === Dice and IoU functions ===
def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice.item()

def iou_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.item()

# === Constants ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "/content/drive/MyDrive/binary_seg_models"
num_classes = 7

# === Store results ===
results = []

# === Loop over all 7 classes ===
for class_id in range(num_classes):
    print(f"\n📊 Evaluating Model for Class {class_id}")

    # === Load validation dataset for current class ===
    val_img = []
    val_msk = []

    for fname in os.listdir(image_dir):
        if fname.endswith("_sat.jpg"):
            image_id = fname.replace("_sat.jpg", "")
            mask_path = os.path.join(binary_mask_dir, f"{image_id}_mask_class_{class_id}.png")
            if os.path.exists(mask_path):
                val_img.append(os.path.join(image_dir, fname))
                val_msk.append(mask_path)

    val_img = val_img[int(0.8 * len(val_img)):]
    val_msk = val_msk[int(0.8 * len(val_msk)):]

    val_ds = BinarySegDataset(val_img, val_msk, transform)
    val_loader = DataLoader(val_ds, batch_size=4)

    # === Load model ===
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    model_path = os.path.join(save_dir, f"binary_segformer_class_{class_id}.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # === Evaluate ===
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
            outputs = torch.sigmoid(outputs)

            for pred, gt in zip(outputs, masks):
                dice_scores.append(dice_score(pred, gt))
                iou_scores.append(iou_score(pred, gt))

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    print(f"✅ Class {class_id}: Dice = {avg_dice:.4f}, IoU = {avg_iou:.4f}")
    results.append((class_id, avg_dice, avg_iou))

# === Optional: Save results as CSV ===
import pandas as pd

results_df = pd.DataFrame(results, columns=["Class", "Dice Score", "IoU"])
results_df.to_csv("/content/drive/MyDrive/binary_seg_eval_results.csv", index=False)

# %%



