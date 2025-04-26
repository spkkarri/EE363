import torch
from transformers import AutoImageProcessor
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import time
from .segformer import SegformerForSemanticSegmentation, get_config

class SegmentationModel:
    def __init__(self, model_path, class_csv_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing on {self.device}...")
        
        # Load components
        self.config = get_config()
        self.model = self._load_model(model_path)
        self.class_rgb_values = pd.read_csv(class_csv_path)[['r','g','b']].values.tolist()
        self.feature_extractor = AutoImageProcessor.from_pretrained(
            "nvidia/mit-b3", size=512, do_normalize=False)
        
        # Warmup
        self._warmup()
    
    def _load_model(self, model_path):
        """Load architecture + weights in one step"""
        model = SegformerForSemanticSegmentation(self.config)
        state_dict = torch.load(model_path, map_location=self.device)
        state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}  # Remove DataParallel prefix
        model.load_state_dict(state_dict)
        return model.to(self.device).eval()
    
    def _warmup(self):
        """Initialize all layers with dummy input"""
        dummy = torch.randn(1, 3, 512, 512).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def predict(self, image_path):
        """Optimized prediction pipeline"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        inputs = self.feature_extractor(
            images=image, 
            return_tensors="pt"
        )['pixel_values'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(inputs)
            pred_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
        
        # Convert to RGB
        rgb_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for idx, color in enumerate(self.class_rgb_values):
            rgb_mask[pred_mask == idx] = color
            
        return image, rgb_mask