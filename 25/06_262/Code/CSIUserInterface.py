<<<<<<< HEAD
# %%
from google.colab import drive
drive.mount('/content/drive')
import torch
import timm  # if you're using Vision Transformer from timm
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
model.load_state_dict(torch.load('/content/drive/MyDrive/vit_eurosat.pth', map_location=torch.device('cpu')))
model.eval()
!pip install gradio
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import timm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model (replace 'vit_base_patch16_224' if different)
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
model.load_state_dict(torch.load('/content/drive/MyDrive/vit_eurosat.pth', map_location=device))
model.to(device)
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Class labels (replace with your actual EuroSAT class names)
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Prediction function
def predict_image(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return class_names[predicted.item()]

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="EuroSAT Image Classifier",
    description="Upload a satellite image to classify its land cover type."
)

# Launch interface
interface.launch()


=======
# %%
from google.colab import drive
drive.mount('/content/drive')
import torch
import timm  # if you're using Vision Transformer from timm
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
model.load_state_dict(torch.load('/content/drive/MyDrive/vit_eurosat.pth', map_location=torch.device('cpu')))
model.eval()
!pip install gradio
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import timm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model (replace 'vit_base_patch16_224' if different)
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
model.load_state_dict(torch.load('/content/drive/MyDrive/vit_eurosat.pth', map_location=device))
model.to(device)
model.eval()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Class labels (replace with your actual EuroSAT class names)
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
               'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Prediction function
def predict_image(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    return class_names[predicted.item()]

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="EuroSAT Image Classifier",
    description="Upload a satellite image to classify its land cover type."
)

# Launch interface
interface.launch()


>>>>>>> 22dc819f6ec92a9a81e9b1b4b4c420395a65edc2
