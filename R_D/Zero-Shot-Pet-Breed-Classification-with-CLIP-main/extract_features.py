import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor
import PIL.Image
from tqdm import tqdm

annotations_dir = "dataset/oxford-iiit-pet/annotations"
images_dir = "dataset/oxford-iiit-pet/images"

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", device)

def parse_txt(file_path):
    entries = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                img_name = parts[0]
                class_id = int(parts[1]) - 1 # Convert to 0-indexed
                entries.append((img_name, class_id))
    return entries

class PreprocessedPetDataset(Dataset):
    def __init__(self, entries, img_dir, processor):
        self.entries = entries
        self.img_dir = img_dir
        self.processor = processor
        
    def __len__(self):
        return len(self.entries)
        
    def __getitem__(self, idx):
        img_name, class_id = self.entries[idx]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        try:
            image = PIL.Image.open(img_path).convert("RGB")
            # Preprocess with CLIPProcessor
            pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
            return pixel_values, class_id
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            # Return dummy tensor and label -1
            return torch.zeros(3, 224, 224), -1

def extract_features(dataset, model):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_pixels, batch_labels in tqdm(loader, desc="Extracting"):
            # Filter out error reads
            valid_mask = batch_labels != -1
            if not valid_mask.any():
                continue
            batch_pixels = batch_pixels[valid_mask].to(device)
            batch_labels = batch_labels[valid_mask].to(device)
            
            # Extract image features
            img_feats = model.get_image_features(pixel_values=batch_pixels)
            # L2 Normalize
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
            features_list.append(img_feats.cpu())
            labels_list.append(batch_labels.cpu())
            
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels

def main():
    print("Loading CLIP Model and Processor...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    trainval_entries = parse_txt(os.path.join(annotations_dir, "trainval.txt"))
    test_entries = parse_txt(os.path.join(annotations_dir, "test.txt"))
    
    trainval_dataset = PreprocessedPetDataset(trainval_entries, images_dir, processor)
    test_dataset = PreprocessedPetDataset(test_entries, images_dir, processor)
    
    print(f"Extracting trainval features ({len(trainval_entries)} entries)...")
    trainval_features, trainval_labels = extract_features(trainval_dataset, model)
    
    print(f"Extracting test features ({len(test_entries)} entries)...")
    test_features, test_labels = extract_features(test_dataset, model)
    
    # Save features
    torch.save(trainval_features, "dataset/trainval_features.pt")
    torch.save(trainval_labels, "dataset/trainval_labels.pt")
    torch.save(test_features, "dataset/test_features.pt")
    torch.save(test_labels, "dataset/test_labels.pt")
    
    print("Extraction completed! Saved to dataset/")

if __name__ == "__main__":
    main()
