import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors

CATS_MAP = {"top":["bottom","shoes"], "bottom":["top","shoes"], "shoes":["top","bottom"]}

class Recommender:
    def __init__(self, emb_path="data/embeddings.npy", meta_path="data/metadata.csv"):
        self.emb = np.load(emb_path).astype(np.float32)
        self.meta = pd.read_csv(meta_path)
        if len(self.emb) != len(self.meta):
            self.meta = pd.read_csv("data/metadata_aligned.csv")
            assert len(self.emb) == len(self.meta), "mismatch embeddings/metadata"
        self.nn = NearestNeighbors(metric="cosine").fit(self.emb)
        if {"color_h", "color_s", "color_v"}.issubset(self.meta.columns):
            self.colors = self.meta[["color_h", "color_s", "color_v"]].to_numpy(dtype=np.float32)
        else:
            self.colors = None
    def color_similarity(self, c1, c2):
        """
        c1, c2 : vecteurs HSV (3,) dans [0,1].

        Retourne une similarité dans [0,1] :
        - 1 = couleurs identiques
        - 0 = très différentes
        """
        h1, s1, v1 = c1
        h2, s2, v2 = c2

        dh = abs(h1 - h2)
        dh = min(dh, 1.0 - dh)

        ds = abs(s1 - s2)
        dv = abs(v1 - v2)

        dist = 0.6 * dh + 0.2 * ds + 0.2 * dv
        sim = 1.0 - dist
        if sim < 0.0:
            sim = 0.0
        if sim > 1.0:
            sim = 1.0
        return float(sim)


    def similar_by_id(self, item_id:int, k:int=5, same_category=True):
        row = self.meta[self.meta["id"]==item_id]
        if row.empty: raise ValueError("id not found")
        idx = row.index[0]
        n = min(k + 1, len(self.emb))
        dist, ind = self.nn.kneighbors([self.emb[idx]], n_neighbors=n)
        out, cat = [], row.iloc[0].get("category", None)
        for i, d in zip(ind[0], dist[0]):
            if i == idx: continue
            r = self.meta.iloc[i].to_dict()
            if same_category and cat and r.get("category")!=cat: continue
            r["score"] = float(1.0 - d); out.append(r)
            if len(out)==k: break
        return out

    def similar_by_vector(self, vec:np.ndarray, k:int=5, category=None):
        n = min(k * 3, len(self.emb))
        dist, ind = self.nn.kneighbors([vec], n_neighbors=n)
        out=[]
        for i, d in zip(ind[0], dist[0]):
            r = self.meta.iloc[i].to_dict()
            if category and r.get("category")!=category: continue
            r["score"] = float(1.0 - d); out.append(r)
            if len(out)==k: break
        return out

    def recommend_outfit(self, item_id: int, per_cat: int = 2):
        row = self.meta[self.meta["id"] == item_id]
        if row.empty:
            raise ValueError(f"item_id {item_id} not found in metadata")

        src_cat = row.iloc[0].get("category", None)
        if src_cat not in CATS_MAP:
            return {}

        idx = row.index[0]
        q = self.emb[idx]
        src_color = self.colors[idx] if getattr(self, "colors", None) is not None else None

        outfit = {}
        for tgt_cat in CATS_MAP[src_cat]:
            mask = (self.meta["category"] == tgt_cat).values
            cand_idx = np.where(mask)[0]
            if len(cand_idx) == 0:
                outfit[tgt_cat] = []
                continue

            cand_vecs = self.emb[cand_idx]
            sims_vis = cand_vecs @ q 

            scores = []
            for j, base_sim in enumerate(sims_vis):
                score = float(base_sim)
                color_sim = None

                if src_color is not None and self.colors is not None:
                    cand_color = self.colors[cand_idx[j]]
                    color_sim = self.color_similarity(src_color, cand_color)
                    score = 0.8 * float(base_sim) + 0.2 * float(color_sim)

                scores.append((score, j, color_sim))

            # on garde les meilleurs
            scores.sort(reverse=True, key=lambda x: x[0])
            keep = scores[:per_cat]

            items = []
            for score, j, color_sim in keep:
                rec_row = self.meta.iloc[cand_idx[j]].to_dict()
                rec_row["score"] = float(score)
                if color_sim is not None:
                    rec_row["color_score"] = float(color_sim)
                    rec_row["color_name"] = rec_row.get("color_name", rec_row.get("color", ""))
                items.append(rec_row)

            outfit[tgt_cat] = items

        return outfit