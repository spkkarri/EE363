<<<<<<< HEAD
# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
from torchvision.datasets import EuroSAT
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import timm
import torch.nn as nn
# %%
import torch
import torch.optim as optim
from tqdm.notebook import tqdm

# %%
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# %%
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

train_dataset = EuroSAT(root="./data", transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# %%
'''import timm
import torch.nn as nn'''

# Swapping classifier head to match 10 EuroSAT classes
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, 10)


# %%
import torch
import torch.optim as optim
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")


# %%
'''from sklearn.metrics import classification_report, confusion_matrix
import numpy as np'''

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))


# %%
#import matplotlib.pyplot as plt

def imshow(img, label, pred):
    img = img.permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + \
          torch.tensor([0.485, 0.456, 0.406])
    img = img.numpy().clip(0, 1)
    plt.imshow(img)
    plt.title(f"True: {label} | Pred: {pred}")
    plt.axis('off')
    plt.show()

# Show few predictions
for i in range(5):
    img, label = train_dataset[i]
    model.eval()
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    imshow(img, train_dataset.classes[label], train_dataset.classes[pred])



=======
# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
from torchvision.datasets import EuroSAT
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import timm
import torch.nn as nn
# %%
import torch
import torch.optim as optim
from tqdm.notebook import tqdm

# %%
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# %%
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean/std
                         std=[0.229, 0.224, 0.225])
])

train_dataset = EuroSAT(root="./data", transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# %%
'''import timm
import torch.nn as nn'''

# Swapping classifier head to match 10 EuroSAT classes
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.head = nn.Linear(model.head.in_features, 10)


# %%
import torch
import torch.optim as optim
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")


# %%
'''from sklearn.metrics import classification_report, confusion_matrix
import numpy as np'''

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))


# %%
#import matplotlib.pyplot as plt

def imshow(img, label, pred):
    img = img.permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + \
          torch.tensor([0.485, 0.456, 0.406])
    img = img.numpy().clip(0, 1)
    plt.imshow(img)
    plt.title(f"True: {label} | Pred: {pred}")
    plt.axis('off')
    plt.show()

# Show few predictions
for i in range(5):
    img, label = train_dataset[i]
    model.eval()
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
    imshow(img, train_dataset.classes[label], train_dataset.classes[pred])



>>>>>>> 22dc819f6ec92a9a81e9b1b4b4c420395a65edc2
