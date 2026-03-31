# FashionMatch - Recommandation de vêtements (API FastAPI)

Projet Python pour préparer un dataset mode, générer des embeddings visuels, puis recommander des articles similaires et des outfits via une API FastAPI.

## Fonctionnalités

- Recommandation d’articles similaires à partir d’un `item_id`
- Recommandation à partir d’une image uploadée
- Suggestion d’outfit par catégories (top / bottom / shoes)
- Extraction de couleurs dominantes (HSV + nom de couleur)
- Endpoint de santé pour vérifier que l’API tourne (`/health`)

## Arborescence

fashionn/
├─ api/  
│  ├─ main.py                 # API FastAPI (routes + chargement modèle)  
│  └─ recommender.py          # logique de recommandation  
├─ scripts/  
│  ├─ prepare_from_kaggle.py  
│  ├─ prepare_from_kaggle_fashion_small.py  
│  ├─ gen_metadata.py  
│  ├─ compute_colors.py  
│  ├─ embed.py  
│  ├─ seed_images.py  
│  └─ seed_fashion_dataset.py  
├─ data/                      # données locales (images, csv, embeddings)  
├─ requirements.txt  
└─ README.md

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
