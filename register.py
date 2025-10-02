# register_person.py
import cv2
import torch
from torchreid.utils import FeatureExtractor
from pymongo import MongoClient
import datetime
import time
import numpy as np

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["person_reid"]
people_col = db["people"]

# Torchreid extractor
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='C:/Users/adity/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

def extract_feature(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Failed to load image {img_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feat = extractor([img_rgb])  # torch tensor
    return feat[0].cpu().numpy()


def register_person(name, role, image_paths):
    features = []
    for path in image_paths:
        feat = extract_feature(path)
        if feat is not None:
            # ensure numpy array
            if isinstance(feat, list):
                feat = np.array(feat)
            features.append(feat)

    if not features:
        print(f"[!] No valid images found for {name}, skipping registration.")
        return

    record = {
        "name": name,
        "role": role,
        "features": [f.flatten().tolist() for f in features if f is not None],
        "registered_at": time.time()
    }

    people_col.insert_one(record)
    print(f"[+] Registered {name} ({role}) with {len(features)} samples")




# Example usage
if __name__ == "__main__":
    register_person("advait", "Employee3", ["samples/adv1.jpeg", "samples/adv2.jpeg", "samples/adv3.jpeg", "samples/adv4.jpeg", "samples/adv5.jpeg", "samples/adv6.jpeg", "samples/adv7.jpeg", "samples/adv8.jpeg", "samples/adv9.jpeg", "samples/adv10.jpeg", "samples/adv11.jpeg", "samples/adv12.jpeg"])

