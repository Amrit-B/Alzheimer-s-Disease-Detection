import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, limit=None):
        """
        Args:
            root_dir (str): Path to ADNI folder.
            limit (int, optional): If set, only loads N images per class (for speed).
        """
        self.data = []
        self.labels = []
        self.class_map = {'CN': 0, 'AD': 1}
        
        for category, label in self.class_map.items():
            dir_path = os.path.join(root_dir, category)
            if not os.path.exists(dir_path): continue
            
            files = sorted([f for f in os.listdir(dir_path) if f.endswith('.nii')])
            
            # Apply limit if requested (e.g., limit=10 for inference)
            if limit:
                files = files[:limit]

            for f_name in files: 
                try:
                    f_path = os.path.join(dir_path, f_name)
                    nii = nib.load(f_path)
                    vol = nii.get_fdata()
                    
                    # Middle Slice Strategy
                    mid = vol.shape[2] // 2
                    slice_2d = vol[:, :, mid]
                    
                    # Preprocessing
                    t = torch.FloatTensor(slice_2d).unsqueeze(0).unsqueeze(0)
                    t = F.interpolate(t, size=(64, 64), mode='bilinear', align_corners=False)
                    t = (t - t.min()) / (t.max() - t.min()) 
                    
                    self.data.append(t.squeeze(0))
                    self.labels.append(label)
                except Exception as e:
                    print(f"Skipping {f_name}: {e}")

    def __len__(self): 
        return len(self.data)
    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx]