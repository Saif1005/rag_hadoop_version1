import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Chemins
DATA_PATH = "/mnt/c/Users/saifa/projet_rag_hadoop/faiss_db/doc_2_fixed.json"
INDEX_PATH = "/mnt/c/Users/saifa/projet_rag_hadoop/faiss_db/index.index"

try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["content"] for item in data]
except Exception as e:
    print(f"Erreur de chargement : {e}")
    exit(1)

# Générer les embeddings avec normalisation
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(texts, normalize_embeddings=True)  # Normalisation L2

# Créer l'index FAISS
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
print("Index FAISS reconstruit.")
