import csv, os, glob

ROWS=[]
cid=0
for cat in ["top","bottom","shoes"]:
    for p in glob.glob(f"data/images/{cat}/*"):
        if os.path.isdir(p): 
            continue
        color = "unknown"  # on pourra l'estimer plus tard
        ROWS.append([cid, p.replace("\\","/"), cat, color])
        cid += 1

os.makedirs("data", exist_ok=True)
with open("data/metadata.csv","w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id","path","category","color"])
    w.writerows(ROWS)

print(f"OK: {len(ROWS)} lignes écrites dans data/metadata.csv")
