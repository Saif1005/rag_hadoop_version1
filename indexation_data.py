import json
from textwrap import wrap

# Chemin des fichiers
input_path = "/mnt/c/Users/saifa/projet_rag_hadoop/faiss_db/doc_2.json"
output_path = "/mnt/c/Users/saifa/projet_rag_hadoop/faiss_db/doc_2_fixed.json"

def split_into_chunks(text, chunk_size=500, overlap=50):
    """Découpe le texte en chunks de taille fixe avec overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# 1. Lire le contenu brut
with open(input_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 2. Découper le texte en chunks cohérents (3 méthodes au choix)
chunks = []
if False:  # Méthode 1: Par paragraphes (déjà présent)
    chunks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
elif True:  # Méthode 2: Par taille fixe (ex: 500 caractères)
    chunks = wrap(raw_text, width=500, break_long_words=False)
else:  # Méthode 3: Par phrases (approximation)
    chunks = [s.strip() for s in raw_text.split('.') if s.strip()]

# 3. Créer une liste d'objets JSON
data = [
    {
        "content": chunk,
        "metadata": {
            "source": "doc_2",
            "chunk_id": f"chunk_{i+1}",
            "length": len(chunk)
        }
    } 
    for i, chunk in enumerate(chunks)
]

# 4. Sauvegarder en JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Document découpé en {len(chunks)} chunks. Fichier sauvegardé sous : {output_path}")
print(f"Exemple du premier chunk : {chunks[0][:100]}...")
