import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Load your CSV manifest
df = pd.read_csv("out/saved_unique_crops/manifest.csv")

# Load CLIP model (from HuggingFace)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Compute embeddings for each image
embeddings = []
for path in tqdm(df["saved_path"], desc="Encoding images"):
    try:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs).squeeze().cpu().numpy()
        # Normalize the vector for cosine similarity
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        embeddings.append(np.zeros(model.config.projection_dim))

embeddings = np.stack(embeddings)

# Compute similarity matrix (cosine similarity)
similarity_matrix = cosine_similarity(embeddings)

# Flag pairs that are very similar (but not identical)
threshold = 0.93  # tweak this; 0.9–0.95 is typical for near-duplicates
pairs = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        sim = similarity_matrix[i, j]
        if sim > threshold:
            pairs.append((df.loc[i, "saved_path"], df.loc[j, "saved_path"], sim))

print("\n🧩 Potential Near-Duplicate Pairs:")
for a, b, sim in sorted(pairs, key=lambda x: -x[2]):
    print(f"{a}  <->  {b}   (similarity={sim:.3f})")
