

import json

def fix_json_format(file_path: str = "faiss_db/doc_2_fixed.json"):
    """Corrige le format du fichier JSON si nécessaire"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Si les données sont une liste, convertir en dict
        if isinstance(data, list):
            data = {str(i): chunk for i, chunk in enumerate(data)}
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("✅ Format corrigé (liste → dict)")

        # Vérifier que tous les chunks sont des strings
        invalid_chunks = [k for k, v in data.items() if not isinstance(v, str)]
        if invalid_chunks:
            print(f"❌ Chunks invalides (IDs): {invalid_chunks}")
            return False

        print("✅ Format valide")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {str(e)}")
        return False

if __name__ == "__main__":
    fix_json_format()
