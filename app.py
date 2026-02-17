import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- Import Model ---
# Ensure src is in python path or located in the same directory
try:
    from src.model import SimpleCNN
except ImportError:
    st.error("Could not import 'src.model'. Make sure the 'src' folder is in the same directory as this app.")
    st.stop()

# --- Config ---
MODEL_PATH = 'weights.pth'
IMG_SIZE = 64
CLASS_NAMES = ['CN (Cognitively Normal)', 'AD (Alzheimer’s Disease)']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Loads the trained model weights."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
        return None
    
    model = SimpleCNN().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return None
        
    model.eval()
    return model

def preprocess_image(uploaded_file):
    """Converts uploaded file to tensor compatible with the model."""
    image = Image.open(uploaded_file).convert('L') # Convert to Grayscale
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    
    # Normalize to [0, 1] if model expects it (assuming standard scaling)
    # If your training used specific mean/std, apply that here.
    img_array = img_array / 255.0
    
    # Convert to tensor: (Batch, Channel, Height, Width)
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor, img_array

def generate_heatmap(model, img_tensor):
    """Generates Grad-CAM heatmap (Adapted from inference.py)."""
    model.eval()
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])
    def forward_hook(module, input, output): activations.append(output)
    
    # Ensure 'conv3' exists in your model, otherwise change this layer name
    if not hasattr(model, 'conv3'):
        return None
        
    handle_b = model.conv3.register_full_backward_hook(backward_hook)
    handle_f = model.conv3.register_forward_hook(forward_hook)
    
    preds = model(img_tensor.to(DEVICE))
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
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - np.min(cam)
    if np.max(cam) != 0: 
        cam = cam / np.max(cam)
    
    handle_b.remove()
    handle_f.remove()
    return cam

# --- Streamlit UI ---
st.title("🧠 Alzheimer’s Disease Detection")
st.write("Upload an MRI slice to detect signs of Alzheimer's Disease.")

model = load_model()

uploaded_file = st.file_uploader("Choose an MRI Image (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # 1. Preprocess
    img_tensor, original_img = preprocess_image(uploaded_file)
    
    # 2. Inference
    with torch.no_grad():
        outputs = model(img_tensor.to(DEVICE))
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

    # 3. Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_img, caption="Uploaded MRI", use_container_width=True, clamp=True)
    
    with col2:
        st.metric(label="Prediction", value=CLASS_NAMES[pred_idx])
        st.metric(label="Confidence", value=f"{confidence:.2%}")
        
        heatmap = generate_heatmap(model, img_tensor)
        if heatmap is not None:
            # Overlay heatmap
            fig, ax = plt.subplots()
            ax.imshow(original_img, cmap='gray')
            ax.imshow(heatmap, cmap='jet', alpha=0.5)
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("Could not generate Grad-CAM (Layer 'conv3' not found).")