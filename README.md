# FashionMatch API

API FastAPI de recommandation mode (tops, bottoms, shoes) avec embeddings visuels (ResNet50), recherche de similarite et suggestion d'outfit.

## Fonctionnalites

- Recommandation d'articles similaires par `item_id`.
- Recommandation d'articles similaires a partir d'une image upload.
- Suggestion d'outfit cross-categories (ex: top -> bottom + shoes).
- Healthcheck pour verifier que l'API est en ligne.

## Structure du projet

- `api/` : code de l'API (`main.py`, `recommender.py`)
- `scripts/` : preparation des donnees et generation des embeddings
- `data/` : donnees locales (images, metadata, embeddings) - **a ne pas versionner en entier**
- `requirements.txt` : dependances Python

## Prerequis

- Python 3.10+ (teste en 3.12)
- Environnement virtuel recommande

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancer l'API

Depuis la racine du projet:

```bash
python -m uvicorn api.main:app --reload
```

API disponible sur:

- `http://127.0.0.1:8000`
- docs Swagger: `http://127.0.0.1:8000/docs`

## Endpoints principaux

- `GET /health`
- `GET /recommend/similar/{item_id}?k=5`
- `GET /recommend/outfit/{item_id}?per_cat=2`
- `POST /recommend/similar` (form-data avec un fichier image, champ `file`)

## Preparation des donnees (optionnel)

Si tu repars de zero:

1) Generer une metadata depuis un dossier d'images:

```bash
python scripts/gen_metadata.py
```

2) Ajouter des infos couleur:

```bash
python scripts/compute_colors.py
```

3) Generer les embeddings:

```bash
python scripts/embed.py --metadata data/metadata.csv --out data/embeddings.npy
```

Tu peux aussi utiliser les scripts Kaggle dans `scripts/prepare_from_kaggle*.py`.

## Notes GitHub

- Ne pas commit `.venv/`, `__pycache__/`, fichiers lourds, ni donnees brutes.
- Idealement: versionner seulement le code + `requirements.txt` + ce `README.md`.
- Si besoin de partager des donnees, preferer un lien externe (Drive/Kaggle/HuggingFace) plutot que Git.

## Etat rapide du projet

Verification locale effectuee:

- Import API OK
- Lancement uvicorn OK
- `GET /health` OK
- endpoint de recommandation par id OK
