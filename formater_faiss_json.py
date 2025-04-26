import json

# Chemin des fichiers
input_path = "/mnt/c/Users/saifa/projet_rag_hadoop/faiss_db/doc_2.json"
output_path = "/mnt/c/Users/saifa/projet_rag_hadoop/faiss_db/doc_2_fixed.json"

# Lire et corriger le fichier
with open(input_path, "r", encoding="utf-8") as f:
    raw_content = f.read().strip()

# Créer une structure valide
fixed_data = [{"content": raw_content, "metadata": {"source": "doc_2"}}]

# Sauvegarder
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(fixed_data, f, ensure_ascii=False, indent=2)

print(f"Fichier corrigé sauvegardé sous : {output_path}")
