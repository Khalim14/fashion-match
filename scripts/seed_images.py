import os, time, random
import requests


CATS = {
    "top":    ["tshirt", "shirt", "blouse", "hoodie", "sweater"],
    "bottom": ["jeans", "trousers", "pants", "skirt"],
    "shoes":  ["sneakers", "shoes", "boots"]
}

def fetch_one(url, dst):
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(dst, "wb") as f:
                f.write(r.content)
            return True
        else:
            print("status", r.status_code, "for", url)
    except Exception as e:
        print("fail:", e)
    return False

def main(n_per_cat=20):
    os.makedirs("data/images/top", exist_ok=True)
    os.makedirs("data/images/bottom", exist_ok=True)
    os.makedirs("data/images/shoes", exist_ok=True)

    random.seed(42)

    for cat, keywords in CATS.items():
        print(f"=== downloading {cat} ===")
        for i in range(n_per_cat):
            q = random.choice(keywords)
            url = f"https://source.unsplash.com/600x800/?{q},clothes,fashion&sig={random.randint(0, 10_000_000)}"
            dst = f"data/images/{cat}/{cat}_{i:03d}.jpg"
            ok = fetch_one(url, dst)
            print(("OK " if ok else "X  "), cat, "->", dst)
            time.sleep(0.2)  

    print("Done.")

if __name__ == "__main__":
    main(20) 
