from PIL import Image
import numpy as np
import os
from datetime import datetime

def generate_unique_filename(filename):
    """Generate unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"

def save_image(image_array, save_path):
    """Save numpy array as image"""
    Image.fromarray(image_array).save(save_path)