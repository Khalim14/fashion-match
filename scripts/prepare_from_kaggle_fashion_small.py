import os
import random
import shutil
import pandas as pd

KAGGLE_DIR = "data/kaggle"
OUT_IMAGES = "data/images"
OUT_META = "data/metadata.csv"

N_TOP = 80
N_BOTTOM = 80
N_SHOES = 80

TOP_TYPES = ["tshirt", "t-shirt", "tee", "shirt", "top", "blouse"]
BOTTOM_TYPES = ["jeans", "trouser", "trousers", "pant", "pants", "short", "shorts"]
SHOES_TYPES = ["shoe", "shoes", "sneaker", "sneakers", "sandal", "sandals"]


def find_images_dir(base):
    for root, dirs, files in os.walk(base):
        if os.path.basename(root).lower() == "images":
            # on vérifie qu'il y a bien des .jpg dedans
            if any(f.lower().endswith(".jpg") for f in files):
                return root
    return None


def find_styles_csv(base):
    for root, dirs, files in os.walk(base):
        for f in files:
            name = f.lower()
            if name.startswith("styles") and name.endswith(".csv"):
                return os.path.join(root, f)
    return None


def map_category(row):
    at = str(row.get("articleType", "")).lower()

    if any(k in at for k in TOP_TYPES):
        return "top"
    if any(k in at for k in BOTTOM_TYPES):
        return "bottom"
    if any(k in at for k in SHOES_TYPES):
        return "shoes"
    return None


def main():
    images_dir = find_images_dir(KAGGLE_DIR)
    styles_csv = find_styles_csv(KAGGLE_DIR)

    if images_dir is None:
        raise RuntimeError(f"Impossible de trouver un dossier 'images' avec des .jpg dans {KAGGLE_DIR}")
    if styles_csv is None:
        raise RuntimeError(f"Impossible de trouver un fichier 'styles*.csv' dans {KAGGLE_DIR}")

    print("images_dir :", images_dir)
    print("styles_csv :", styles_csv)

    # créer les dossiers de sortie
    for cat in ["top", "bottom", "shoes"]:
        os.makedirs(os.path.join(OUT_IMAGES, cat), exist_ok=True)

    print("Lecture du styles.csv…")
    df = pd.read_csv(
        styles_csv,
        engine="python",    
        on_bad_lines="skip"  
    )

    print("Mapping des catégories…")
    df["cat"] = df.apply(map_category, axis=1)
    df = df.dropna(subset=["cat"])

    import numpy as np
    print("Nombre total d'articles top/bottom/shoes :", len(df))

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
            src = os.path.join(images_dir, f"{idv}.jpg")
            if not os.path.exists(src):
                continue

            dst_rel = os.path.join("data/images", cat, f"{idv}.jpg").replace("\\", "/")
            dst_abs = os.path.join(OUT_IMAGES, cat, f"{idv}.jpg")

            if not os.path.exists(dst_abs):
                shutil.copyfile(src, dst_abs)

            rows.append([cid, dst_rel, cat, "unknown"])
            cid += 1

    if not rows:
        raise RuntimeError("Aucune image n'a été copiée. Vérifie le dataset Kaggle.")

    meta = pd.DataFrame(rows, columns=["id", "path", "category", "color"])
    meta.to_csv(OUT_META, index=False)
    print(f"OK: {len(meta)} lignes écrites dans {OUT_META}")


if __name__ == "__main__":
    main()
