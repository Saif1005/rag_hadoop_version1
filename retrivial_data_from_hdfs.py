import os
import json
import numpy as np
import faiss
import spacy
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Any

# ===== CONFIGURATION =====
load_dotenv()
FAISS_DB_PATH = os.getenv('FAISS_DB_PATH', 'faiss_db/index.index')
GOAL_DATA_DIR = os.getenv('GOAL_DATA_DIR', 'faiss_db')

# Modèle spaCy pour la segmentation
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    spacy.cli.download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

# ===== FONCTIONS PRINCIPALES =====
def split_chunks(text: str, max_chunk_size: int = 300) -> List[str]:
    """Découpe un texte en chunks basés sur des phrases."""
    doc = nlp(text)
    chunks = []
    current_chunk = ""

    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        if len(current_chunk) + len(sentence_text) <= max_chunk_size:
            current_chunk += " " + sentence_text if current_chunk else sentence_text
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence_text
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def get_embedding_function():
    """Retourne le modèle d'embedding multilingue."""
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def add_to_faiss(chunks: List[str], embedder: SentenceTransformer, save_path: str = FAISS_DB_PATH) -> Dict[str, Any]:
    """Ajoute les embeddings des chunks à l'index FAISS."""
    if not chunks:
        return {"status": "error", "message": "Aucun chunk à indexer."}

    try:
        print("🔹 Génération des embeddings...")
        embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Vérification des embeddings
        if np.isnan(embeddings).any():
            raise ValueError("Embeddings contiennent des valeurs NaN")
            
        print("🔹 Normalisation L2...")
        faiss.normalize_L2(embeddings)
        
        print("🔹 Création de l'index FAISS...")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(index, save_path)
        
        return {"status": "success", "message": f"{len(chunks)} chunks indexés."}
        
    except Exception as e:
        return {"status": "error", "message": f"Erreur lors de l'indexation: {str(e)}"}

def save_chunks_to_json(chunks: List[str], goal_id: str) -> bool:
    """Sauvegarde les chunks dans un fichier JSON."""
    try:
        json_path = os.path.join(GOAL_DATA_DIR, f"{goal_id}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        chunks_dict = {str(i): chunk for i, chunk in enumerate(chunks)}
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {str(e)}")
        return False

def verify_data(goal_id: str = "doc_2_fixed"):
    """Vérifie et corrige le format des données"""
    try:
        with open(f"{GOAL_DATA_DIR}/{goal_id}.json", "r") as f:
            data = json.load(f)
        
        # Conversion liste → dict si nécessaire
        if isinstance(data, list):
            data = {str(i): chunk for i, chunk in enumerate(data)}
            with open(f"{GOAL_DATA_DIR}/{goal_id}.json", "w") as f:
                json.dump(data, f, indent=2)
        
        # Vérification du contenu
        if not data or len(data) == 0:
            raise ValueError("Le fichier JSON est vide")
            
        first_chunk = list(data.values())[0]
        if not isinstance(first_chunk, str) or len(first_chunk) < 20:
            raise ValueError("Format de chunk invalide")
            
        print(f"✅ Données vérifiées ({len(data)} chunks valides)")
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans les données: {str(e)}")
        return False

def enhanced_similarity_search(
    query_text: str,
    goal_id: str,
    top_k: int = 5,
    min_score: float = 0.15,  # Seuil très bas
    max_distance: float = 0.99  # Presque tout accepter
) -> List[Dict[str, Any]]:
    try:
        # Chargement des données avec vérification
        if not verify_data(goal_id):
            return []

        # 1. Vérification du fichier de données
        json_path = os.path.join(GOAL_DATA_DIR, f"{goal_id}.json")
        if not os.path.exists(json_path):
            print(f"❌ Fichier {json_path} introuvable.")
            return []

        # 2. Chargement des chunks
        with open(json_path, "r", encoding='utf-8') as f:
            chunks_data = json.load(f)
            
        if not chunks_data:
            print("⚠️ Aucun chunk disponible dans le fichier.")
            return []

        # 3. Vérification de l'index FAISS
        if not os.path.exists(FAISS_DB_PATH):
            print(f"❌ Index FAISS introuvable : {FAISS_DB_PATH}")
            return []

        # 4. Génération de l'embedding de la requête
        model = get_embedding_function()
        query_embedding = model.encode([query_text], convert_to_numpy=True)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        # 5. Recherche dans l'index FAISS
        index = faiss.read_index(FAISS_DB_PATH)
        scores, indices = index.search(query_embedding, top_k * 2)  # Recherche élargie
        
        # 6. Traitement des résultats
        if len(scores) == 0 or len(indices) == 0:
            print("⚠️ Aucun résultat retourné par FAISS.")
            return []

        # Nouvelle stratégie de scoring
        scores = np.clip(scores, -1, 1)
        adjusted_scores = (scores + 1) / 2  # Conversion [-1,1] → [0,1]
        
        results = []
        for score, idx in zip(adjusted_scores[0], indices[0]):
            chunk_id = str(idx)
            if chunk_id in chunks_data and score >= min_score:
                results.append({
                    "text": chunks_data[chunk_id],
                    "similarity_score": float(score),
                    "metadata": {"source": goal_id, "chunk_id": chunk_id}
                })

        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)[:top_k]

    except Exception as e:
        print(f"❌ Erreur critique: {str(e)}")
        return []

def list_available_goals() -> List[str]:
    """Liste les goal_ids disponibles."""
    if not os.path.exists(GOAL_DATA_DIR):
        return []
    return [f.replace('.json', '') for f in os.listdir(GOAL_DATA_DIR) if f.endswith('.json')]

def rebuild_index(goal_id: str = "doc_2_fixed"):
    """Reconstruit complètement l'index FAISS"""
    try:
        # 1. Charger les données
        with open(f"{GOAL_DATA_DIR}/{goal_id}.json", "r") as f:
            chunks_data = json.load(f)
        
        # 2. Vérifier le format
        if isinstance(chunks_data, list):
            chunks_data = {str(i): chunk for i, chunk in enumerate(chunks_data)}
        
        chunks = list(chunks_data.values())
        
        # 3. Générer les embeddings
        embedder = get_embedding_function()
        result = add_to_faiss(chunks, embedder)
        
        if result["status"] == "success":
            print("✅ Index reconstruit avec succès")
            return True
        else:
            print(f"❌ Erreur: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la reconstruction: {str(e)}")
        return False

# ===== TESTS =====
def test_enhanced_search():
    """Teste la recherche améliorée."""
    print("\n=== TEST DE RECHERCHE ===")
    try:
        # Régénérer l'index si absent
        if not os.path.exists(FAISS_DB_PATH):
            print("⚙️ Index FAISS absent. Régénération en cours...")
            embedder = get_embedding_function()
            with open(f"{GOAL_DATA_DIR}/doc_2_fixed.json", "r") as f:
                chunks_data = json.load(f)
            # Convertir en dictionnaire si c'est une liste
            if isinstance(chunks_data, list):
                chunks_data = {str(i): chunk for i, chunk in enumerate(chunks_data)}
            chunks = list(chunks_data.values())
            add_to_faiss(chunks, embedder)  # Recrée l'index
            print("✅ Index FAISS régénéré.")
        
        results = enhanced_similarity_search(
            query_text="défis éthiques IA",
            goal_id="doc_2_fixed",
            min_score=0.1,  # Seuil très bas pour debug
            max_distance=1.0
        )
        
        if not results:
            print("Aucun résultat trouvé.")
            return False
        
        print(f"Résultats obtenus : {len(results)}")
        for i, res in enumerate(results, 1):
            print(f"\nRésultat #{i}:")
            print(f"Score : {res['similarity_score']:.2f}")
            print(f"Contenu : {res['text'][:100]}...")
        return True
    
    except Exception as e:
        print(f"Échec du test : {str(e)}")
        return False

# ===== POINT D'ENTRÉE =====
if __name__ == "__main__":
    print("=== SYSTÈME DE RECHERCHE RAG ===")
    
    # 1. Vérifier et reconstruire l'index si nécessaire
    if not os.path.exists(FAISS_DB_PATH):
        print("⚙️ Reconstruction de l'index FAISS...")
        if not rebuild_index():
            exit(1)
    
    # 2. Exécuter la recherche
    results = enhanced_similarity_search(
        query_text="éthique intelligence artificielle",
        goal_id="doc_2_fixed",
        min_score=0.2,
        max_distance=0.9
    )
    
    if results:
        print("\n🔍 RÉSULTATS :")
        for i, res in enumerate(results, 1):
            print(f"\n#{i} [Score: {res['similarity_score']:.2f}]")
            print(res["text"][:200] + ("..." if len(res["text"]) > 200 else ""))
    else:
        print("\n⚠️ Aucun résultat. Suggestions supplémentaires:")
        print("- Vérifiez que le fichier JSON contient du texte pertinent")
        print("- Essayez d'autres requêtes plus simples")
        print("- Inspectez les logs de construction de l'index")

