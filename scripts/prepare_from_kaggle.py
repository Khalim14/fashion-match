import os
import shutil
import random
import pandas as pd

# Dossiers
KAGGLE_DIR = "data/kaggle"
IMAGES_DIR = os.path.join(KAGGLE_DIR, "images")
STYLES_CSV = os.path.join(KAGGLE_DIR, "styles.csv")

OUT_IMAGES = "data/images"
OUT_META = "data/metadata.csv"

# combien d'images max par catégorie
N_TOP = 80
N_BOTTOM = 80
N_SHOES = 80

def map_category(row):
    mc = str(row.get("masterCategory", "")).lower()
    sc = str(row.get("subCategory", "")).lower()
    at = str(row.get("articleType", "")).lower()


    if mc == "apparel" and any(k in at for k in [
        "shirt", "tshirt", "top", "kurta", "sweat", "hoodie", "blouse", "polo"
    ]):
        return "top"

    if mc == "apparel" and any(k in at for k in [
        "jeans", "trouser", "pant", "short", "skirt", "jogger", "chinos", "leggings"
    ]):
        return "bottom"

    if mc == "footwear":
        return "shoes"

    return None

def main():
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(STYLES_CSV):
        raise RuntimeError("Assure-toi que data/kaggle contient 'images/' et 'styles.csv'")

    for cat in ["top", "bottom", "shoes"]:
        os.makedirs(os.path.join(OUT_IMAGES, cat), exist_ok=True)

    df = pd.read_csv(STYLES_CSV, encoding="utf-8")
    df["cat"] = df.apply(map_category, axis=1)
    df = df.dropna(subset=["cat"])

    random.seed(42)

    rows = []
    cid = 0

    for cat, n_max in [("top", N_TOP), ("bottom", N_BOTTOM), ("shoes", N_SHOES)]:
        df_cat = df[df["cat"] == cat]
        ids = df_cat["id"].tolist()
        random.shuffle(ids)
        ids = ids[:min(n_max, len(ids))]

        print(f"{cat}: {len(ids)} images sélectionnées")

        for idv in ids:
            src = os.path.join(IMAGES_DIR, f"{idv}.jpg")
            if not os.path.exists(src):
                continue
            dst_rel = os.path.join("data/images", cat, f"{idv}.jpg").replace("\\", "/")
            dst_abs = os.path.join(OUT_IMAGES, cat, f"{idv}.jpg")
            if not os.path.exists(dst_abs):
                shutil.copyfile(src, dst_abs)
            rows.append([cid, dst_rel, cat, "unknown"])
            cid += 1

    meta = pd.DataFrame(rows, columns=["id", "path", "category", "color"])
    meta.to_csv(OUT_META, index=False)
    print(f"OK: {len(meta)} lignes écrites dans {OUT_META}")

if __name__ == "__main__":
    main()
