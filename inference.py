import os
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch

# --- Professional Imports ---
from src.model import SimpleCNN
from src.dataset import MedicalImageDataset

def generate_heatmap(model, img_tensor):
    """Generates Grad-CAM heatmap for a given image tensor."""
    model.eval()
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])
    def forward_hook(module, input, output): activations.append(output)
        
    handle_b = model.conv3.register_full_backward_hook(backward_hook)
    handle_f = model.conv3.register_forward_hook(forward_hook)
    
    preds = model(img_tensor)
    score = preds[:, preds.argmax()]
    
    model.zero_grad()
    score.backward()
    
    grads = gradients[0].cpu().data.numpy()[0]
    fmap = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights): 
        cam += w * fmap[i]
        
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (64, 64))
    cam = cam - np.min(cam)
    if np.max(cam) != 0: 
        cam = cam / np.max(cam)
    
    handle_b.remove()
    handle_f.remove()
    return cam

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model_path = 'weights.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
        
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f" Loaded weights from {model_path}")

    # 2. Load Data (Limit to 10 for speed)
    print(" Loading sample data...")
    dataset = MedicalImageDataset('./ADNI', limit=10)
    
    if len(dataset) > 0:
        idx = random.randint(0, len(dataset)-1)
        img_tensor, label = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # 3. Run Inference
        heatmap = generate_heatmap(model, img_tensor)
        
        # 4. Save Result
        class_names = ['CN', 'AD']
        pred_idx = model(img_tensor).argmax().item()
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1); plt.imshow(img_tensor.cpu().squeeze(), cmap='gray'); plt.axis('off'); plt.title(f"True: {class_names[label]}")
        plt.subplot(1, 3, 2); plt.imshow(heatmap, cmap='jet'); plt.axis('off'); plt.title("Grad-CAM")
        plt.subplot(1, 3, 3); plt.imshow(img_tensor.cpu().squeeze(), cmap='gray'); plt.imshow(heatmap, cmap='jet', alpha=0.5); plt.axis('off'); plt.title(f"Pred: {class_names[pred_idx]}")
        
        output_file = 'gradcam_result.png'
        plt.savefig(output_file)
        print(f" Result saved to {output_file}")
    else:
        print(" No data found in ./ADNI")