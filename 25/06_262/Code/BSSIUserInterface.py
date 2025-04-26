# %%
from google.colab import drive
drive.mount('/content/drive')

!pip install gradio transformers --quiet

import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# === Set Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DeepGlobe Class Names ===
class_names = [
    "Urban",         # class 0
    "Agriculture",   # class 1
    "Rangeland",     # class 2
    "Forest",        # class 3
    "Water",         # class 4
    "Barren",        # class 5
    "Unknown"        # class 6
]

# === Path to folder containing 7 model .pth files ===
model_dir = "/content/drive/MyDrive/binary_seg_models"  # update with your path

# === Load all 7 models ===
models = []
for i in range(7):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    model_path = os.path.join(model_dir, f"binary_segformer_class_{i}.pth")  # filenames: binaryseg_class_0.pth, ...
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    models.append(model)

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Prediction function ===
def predict_all_masks(uploaded_image):
    image = uploaded_image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    output_masks = []
    with torch.no_grad():
        for model in models:
            logits = model(pixel_values=image_tensor).logits
            logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
            probs = torch.sigmoid(logits)
            mask = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask).convert("L")
            output_masks.append(mask_pil)

    return output_masks

# === Gradio Interface ===
interface = gr.Interface(
    fn=predict_all_masks,
    inputs=gr.Image(type="pil", label="Upload Satellite Image"),
    outputs=[
        gr.Image(type="pil", label=f"{name} Mask") for name in class_names
    ],
    title="🌍 DeepGlobe Land Cover Segmentation (7-Class)",
    description="Upload a satellite image to get binary segmentation masks for each land cover class using 7 trained SegFormer models."
)

interface.launch()


