import os
import time
import random
import requests

CATS = ["top", "bottom", "shoes"]

BASE_URL = "https://picsum.photos/seed/{seed}/400/600"

def download_image(url: str, dst: str) -> bool:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            with open(dst, "wb") as f:
                f.write(r.content)
            return True
        print("status", r.status_code, "for", url)
    except Exception as e:
        print("error", e, "for", url)
    return False

def main(n_per_cat: int = 30):
    for cat in CATS:
        os.makedirs(f"data/images/{cat}", exist_ok=True)

    random.seed(42)

    for cat in CATS:
        print(f"=== downloading {cat} ===")
        count = 0
        while count < n_per_cat:
            seed = f"{cat}_{count}_{random.randint(0, 10_000_000)}"
            url = BASE_URL.format(seed=seed)
            dst = f"data/images/{cat}/{cat}_{count:03d}.jpg"
            ok = download_image(url, dst)
            if ok:
                print("OK ", dst)
                count += 1
            else:
                print("X   ", dst)
            time.sleep(0.2)

    print("Done.")

if __name__ == "__main__":
    main(20)  
