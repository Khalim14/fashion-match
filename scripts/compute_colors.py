import os
import colorsys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans


def dominant_color(path, k=3):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    img = Image.open(path).convert("RGB")
    w, h = img.size
    crop_w, crop_h = int(w * 0.8), int(h * 0.8)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    img = img.crop((left, top, left + crop_w, top + crop_h))

    img = img.resize((80, 80))

    arr = np.array(img, dtype=np.float32) / 255.0
    pixels = arr.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init=3, random_state=0)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels)

    order = np.argsort(-counts)

    for idx in order:
        r, g, b = centers[idx]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        if s < 0.25 and v > 0.85:
            continue
        if v < 0.1:
            continue

        return float(h), float(s), float(v)

    r, g, b = centers[order[0]]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return float(h), float(s), float(v)

def hsv_to_name(h, s, v):

    if s < 0.2 and v > 0.8:
        return "white"
    if v < 0.2:
        return "black"
    if s < 0.25:
        return "grey"

    deg = h * 360.0

    if deg < 15 or deg >= 345:
        return "red"
    if deg < 45:
        return "orange"
    if deg < 75:
        return "yellow"
    if deg < 150:
        return "green"
    if deg < 210:
        return "cyan"
    if deg < 270:
        return "blue"
    if deg < 300:
        return "purple"
    return "pink"


def main():
    meta_path = "data/metadata.csv"
    if not os.path.exists(meta_path):
        raise RuntimeError("data/metadata.csv introuvable, lance d'abord prepare_from_kaggle + gen_metadata.")

    meta = pd.read_csv(meta_path)

    hs, ss, vs, names = [], [], [], []

    for i, row in meta.iterrows():
        path = row["path"]
        try:
            h, s, v = dominant_color(path)
            name = hsv_to_name(h, s, v)
        except Exception as e:
            print(f"[WARN] couleur impossible pour {path} -> {e}")
            h, s, v, name = 0.0, 0.0, 0.0, "unknown"

        hs.append(h)
        ss.append(s)
        vs.append(v)
        names.append(name)

        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(meta)} images traitées…")

    meta["color_h"] = hs
    meta["color_s"] = ss
    meta["color_v"] = vs
    meta["color_name"] = names

    meta.to_csv(meta_path, index=False)
    print(f"Couleurs ajoutées et metadata réécrit dans {meta_path}")


if __name__ == "__main__":
    main()
