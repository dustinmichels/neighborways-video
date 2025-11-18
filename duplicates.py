import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.types import ImgRecord

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
threshold = 0.90  # tweak this; 0.9–0.95 is typical for near-duplicates


# Build a graph using Union-Find to group similar images
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


# Initialize Union-Find
uf = UnionFind(len(df))

# Find all similar pairs and union them
pairs = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        sim = similarity_matrix[i, j]
        if sim > threshold:
            uf.union(i, j)
            pairs.append((i, j, sim))

# Group images by their root parent (bin)
bins_dict = defaultdict(list)
for i in range(len(df)):
    root = uf.find(i)
    bins_dict[root].append(i)

# Convert to list of bins with ImgRecord objects
bins = []
for indices in bins_dict.values():
    # Sort indices to maintain original timestamp order
    indices.sort()
    bin_records = [
        ImgRecord(
            saved_path=df.loc[idx, "saved_path"],
            label=df.loc[idx, "label"],
            track_id=int(df.loc[idx, "track_id"]),
            frame_no=int(df.loc[idx, "frame_no"]),
            conf=float(df.loc[idx, "conf"]),
        )
        for idx in indices
    ]
    bins.append(bin_records)

# Sort bins by the first record's index to maintain overall order
bins.sort(key=lambda b: df[df["saved_path"] == b[0].saved_path].index[0])

# Convert to JSON-serializable format
output = {
    "bins": [[record.model_dump() for record in bin_records] for bin_records in bins]
}

# Save to JSON file
output_path = "out/duplicate_bins.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to {output_path}")
print("📊 Summary:")
print(f"  - Total files: {len(df)}")
print(f"  - Total bins: {len(bins)}")
print(f"  - Bins with duplicates: {sum(1 for b in bins if len(b) > 1)}")
print(f"  - Singleton bins: {sum(1 for b in bins if len(b) == 1)}")
print(f"  - Duplicate pairs found: {len(pairs)}")

# Optional: Print some example bins with duplicates
print("\n🧩 Example duplicate groups:")
for i, bin_records in enumerate(bins[:5]):  # Show first 5 bins
    if len(bin_records) > 1:
        print(f"\nBin {i + 1} ({len(bin_records)} files):")
        for record in bin_records:
            print(
                f"  - {record.saved_path} (track_id={record.track_id}, frame={record.frame_no})"
            )
