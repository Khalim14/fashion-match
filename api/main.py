from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

from .recommender import Recommender

app = FastAPI(title="FashionMatch API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="data/images"), name="images")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

@torch.no_grad()
def load_backbone():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = torch.nn.Identity()
    m.eval()
    return m

model = load_backbone()
reco = None

def ensure_reco():
    global reco
    if reco is None:
        reco = Recommender("data/embeddings.npy", "data/metadata.csv")
    return reco

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/recommend/similar/{item_id}")
def similar_by_id(item_id: int, k: int = 5):
    try:
        items = ensure_reco().similar_by_id(item_id, k)
        return {"results": items}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/recommend/outfit/{item_id}")
def outfit(item_id: int, per_cat: int = 2):
    try:
        items = ensure_reco().recommend_outfit(item_id, per_cat)
        return {"outfit": items}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/recommend/similar")
async def similar_upload(file: UploadFile = File(...), k: int = 5, category: str | None = None):
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(400, "invalid image")

    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        v = model(x).numpy().squeeze()
        v = v / (np.linalg.norm(v) + 1e-10)

    items = ensure_reco().similar_by_vector(v, k, category)
    return JSONResponse({"results": items})
