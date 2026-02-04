import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
import os

# --- Import from src folder ---
from src.model import SimpleCNN
from src.dataset import MedicalImageDataset

# --- HYPERPARAMETERS ---
IMG_SIZE = 64
BATCH_SIZE = 16 
LR = 0.001
EPOCHS = 20

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading Dataset...")
    full_dataset = MedicalImageDataset('./ADNI')
    
    if len(full_dataset) == 0:
        print("❌ Error: No images found. Did you run setup_data.py?")
        return

    # 2. Split
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    print(f"Split Sizes -> Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # 3. Loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

    # 4. Init Model
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf') 
    
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'weights.pth')
            print(f"    -> Saved 'weights.pth' (Val Loss: {best_val_loss:.4f})")

    print("\n✅ Training Complete. Model saved as 'weights.pth'")

if __name__ == "__main__":
    train_model()