# Retrieval-Augmented Generation (RAG)

La RAG (Retrieval-Augmented Generation) est une approche d'intelligence artificielle qui combine la recherche de documents pertinents (retrieval) et la génération de texte (generation) à l'aide de modèles de langage. Elle permet de fournir des réponses précises et contextualisées en s'appuyant à la fois sur des bases de connaissances et sur la puissance des modèles génératifs.

# Projet RAG Hadoop – Indexation et Recherche de Documents

Ce projet permet d'indexer des documents stockés sur HDFS (Hadoop Distributed File System) dans une base vectorielle ChromaDB, puis de rechercher efficacement des passages pertinents à l'aide de modèles d'embeddings et de similarité sémantique. Il permet également de générer des réponses synthétiques à l'aide d'un modèle de langage avancé comme GPT-3.5-turbo (OpenAI).

## Fonctionnalités principales

- **Extraction de documents depuis HDFS** (par goal_id)
- **Découpage intelligent** des documents en chunks pour l'indexation
- **Indexation vectorielle** dans ChromaDB avec génération d'embeddings (Sentence Transformers)
- **Recherche sémantique** : retrouver les passages les plus similaires à une requête utilisateur
- **Génération de texte** : produire des réponses à partir des documents retrouvés, en utilisant un modèle de langage avancé comme `gpt-3.5-turbo` (OpenAI)
- **API Python** pour automatiser l'indexation et la recherche

## Prérequis

- Python 3.8+
- Un cluster Hadoop opérationnel (HDFS accessible)
- ChromaDB
- Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2`)
- (Optionnel) FastAPI pour exposer une API

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configurer les variables d'environnement**
   - Crée un fichier `.env` à la racine :
     ```
     CHROMA_DB_PATH=chroma_db
     ```

## Utilisation

### 1. Indexer les documents HDFS dans ChromaDB

Lance le script principal pour indexer tous les documents présents dans `/user/hadoop/goals/` sur HDFS :
```bash
python retrivial_data_from_hdfs.py
```

### 2. Rechercher des documents par goal_id

Utilise la fonction `retrieve_documents_by_goal_id(goal_id)` pour récupérer tous les chunks indexés pour un goal_id donné.

### 3. Recherche sémantique

Utilise la fonction `retrieve_similar_chunks(goal_id, query_text, k=5)` pour obtenir les passages les plus proches d'une requête.

### 4. Génération de texte avec GPT-3.5-turbo

Après avoir retrouvé les passages pertinents via la recherche sémantique, le projet permet de générer des réponses synthétiques ou des résumés à l'aide d'un modèle de langage avancé comme `gpt-3.5-turbo` (OpenAI).

**Remarque** : Il faut une clé API OpenAI valide pour utiliser ce service.

## Structure du projet

- `retrivial_data_from_hdfs.py` : Script principal pour l'indexation et la recherche
- `chroma_db/` : Dossier contenant la base vectorielle ChromaDB
- `requirements.txt` : Dépendances Python

## Exemple de flux

1. **Extraction** : Les documents sont extraits de HDFS par goal_id.
2. **Découpage** : Les documents sont découpés en chunks.
3. **Indexation** : Les chunks sont vectorisés et stockés dans ChromaDB.
4. **Recherche** : À partir d'une requête, on retrouve les passages les plus pertinents.
5. **Génération** : Le contexte est envoyé à GPT-3.5-turbo pour générer une réponse.

## Fonctionnement du script `query_data.py`

Le fichier `query_data.py` orchestre tout le pipeline RAG (Retrieval-Augmented Generation) :

1. **Récupération du contexte pertinent** :
   - Utilise la fonction `get_cached_context` pour extraire, à partir de ChromaDB, les passages les plus proches de la question posée (requête utilisateur) pour un `goal_id` donné.
   - Cette étape s'appuie sur la recherche sémantique (embeddings) pour ne retenir que les passages les plus pertinents.
2. **Gestion du cache** :
   - Les réponses générées sont mises en cache (en mémoire) pour accélérer les requêtes répétées et éviter des appels inutiles à l'API OpenAI.
3. **Génération de la réponse** :
   - Si le contexte est jugé pertinent, le script construit un prompt structuré (voir `PROMPT_TEMPLATE`) et l'envoie au modèle `gpt-3.5-turbo` via l'API OpenAI.
   - Le modèle génère une réponse synthétique basée uniquement sur le contexte extrait.
   - Si le contexte n'est pas pertinent ou absent, une réponse standard est retournée.
4. **Utilisation en ligne de commande** :
   - Le script peut être lancé en CLI pour poser une question à un goal_id spécifique, ajuster le nombre de passages utilisés (`top_k`), le seuil de similarité, et afficher les chunks similaires pour debug.

**Exemple d'utilisation en ligne de commande**
```bash
python query_data.py "Qu'est-ce que l'intelligence artificielle ?" --goal_id saifsaif1005 --top_k 5
```

## Détail de la phase de retrieval (récupération de passages)

La fonction clé pour la récupération des passages pertinents dans ce projet est :

### `retrieve_similar_chunks`

```python
def retrieve_similar_chunks(goal_id: str, query_text: str, k: int = 5, score_threshold: float = 1.0, USE_CHROMA: bool = True):
    if not USE_CHROMA : 
        return []
    try:
        if os.path.exists(CHROMA_PATH):
            embedding_function = get_embedding_function()
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collections = client.list_collections()
            if goal_id not in collections:
                return []
            collection = client.get_collection(goal_id)
            query_embedding = embedding_function.encode([query_text], convert_to_numpy=True).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=k)
            ids = results["ids"][0]
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            output = []
            for doc, meta, dist in zip(docs, metadatas, distances):
                output.append({
                    "document": doc,
                    "metadata": meta,
                    "similarity": 1 - dist,
                    "distance": dist
                })
            return output
        else:
            return []
    except Exception as e:
        print(f"Erreur lors de la recherche dans Chroma: {str(e)}")
        return []
```

**Explication** :
- Cette fonction retrouve les passages (chunks) les plus similaires à une requête utilisateur (`query_text`) dans la base vectorielle ChromaDB, pour un `goal_id` donné.
- Elle encode la requête en vecteur, interroge ChromaDB, et retourne les `k` passages les plus proches avec leur score de similarité.
- C'est le cœur de la phase de retrieval du pipeline RAG : elle fournit le contexte pertinent qui sera utilisé pour la génération de réponse par le modèle de langage.

## Auteurs

- Saif (github.com/Saif1005) 