import argparse, os, numpy as np, pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@torch.no_grad()
def load_backbone():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    m.eval()
    return m

@torch.no_grad()
def embed_image(model, path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    v = model(x).numpy().squeeze()          
    v = v / (np.linalg.norm(v) + 1e-10)     # 
    return v.astype(np.float32)

def main(meta_path, out_path):
    meta = pd.read_csv(meta_path)
    model = load_backbone()
    embs, keep = [], []
    for _, row in meta.iterrows():
        p = row["path"]
        if not os.path.exists(p):
            print(f"[WARN] missing: {p}")
            continue
        embs.append(embed_image(model, p))
        keep.append(row["id"])
    embs = np.stack(embs, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, embs)
    print("saved:", embs.shape, "->", out_path)
    meta[meta["id"].isin(keep)].to_csv("data/metadata_aligned.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--out", default="data/embeddings.npy")
    args = ap.parse_args()
    main(args.metadata, args.out)
