import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace

# Automatically resolve dataset path (works even if run via manage.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder where this script lives
dataset_path = os.path.join(BASE_DIR, "dataset")

encodings = []
names = []

print("🔍 Encoding faces...")

# Check dataset exists
if not os.path.exists(dataset_path):
    print(f"❌ Dataset folder not found at: {dataset_path}")
    exit(1)

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"👤 Processing: {person_name}")
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        try:
            # Extract embeddings with DeepFace
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            encodings.append(embedding)
            names.append(person_name)

        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

# Save encodings
data = {"encodings": encodings, "names": names}
with open(os.path.join(BASE_DIR, "encodings.pickle"), "wb") as f:
    pickle.dump(data, f)

print("✅ Encodings generated and saved to encodings.pickle")
