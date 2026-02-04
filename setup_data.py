import os
import shutil
import argparse

def organize_dataset(source_path):
    target_root = "./ADNI"
    required_classes = ['AD', 'CN']
    
    if not os.path.exists(target_root):
        os.makedirs(target_root)

    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' not found.")
        return

    found_count = 0
    for root, dirs, files in os.walk(source_path):
        current_folder = os.path.basename(root)
        
        if current_folder in required_classes:
            target_class_dir = os.path.join(target_root, current_folder)
            
            if not os.path.exists(target_class_dir):
                os.makedirs(target_class_dir)

            print(f"Processing: {current_folder}")
            
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(target_class_dir, file)
                    
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)
            
            found_count += 1

    if found_count == 0:
        print("Warning: No 'AD' or 'CN' folders found.")
    else:
        print(f"Dataset ready in: {os.path.abspath(target_root)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    args = parser.parse_args()
    
    organize_dataset(args.source)