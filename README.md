# Projet RAG Hadoop – Indexation et Recherche de Documents

Ce projet permet d'indexer des documents stockés sur HDFS (Hadoop Distributed File System) dans une base vectorielle ChromaDB, puis de rechercher efficacement des passages pertinents à l'aide de modèles d'embeddings et de similarité sémantique.

## Fonctionnalités principales

- **Extraction de documents depuis HDFS** (par goal_id)
- **Découpage intelligent** des documents en chunks pour l'indexation
- **Indexation vectorielle** dans ChromaDB avec génération d'embeddings (Sentence Transformers)
- **Recherche sémantique** : retrouver les passages les plus similaires à une requête utilisateur
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

### Indexer les documents HDFS dans ChromaDB

Lance le script principal pour indexer tous les documents présents dans `/user/hadoop/goals/` sur HDFS :

```bash
python retrivial_data_from_hdfs.py
```

### Rechercher des documents par goal_id

Utilise la fonction `retrieve_documents_by_goal_id(goal_id)` pour récupérer tous les chunks indexés pour un goal_id donné.

### Recherche sémantique

Utilise la fonction `retrieve_similar_chunks(goal_id, query_text, k=5)` pour obtenir les passages les plus proches d'une requête.

## Structure du projet

- `retrivial_data_from_hdfs.py` : Script principal pour l'indexation et la recherche
- `chroma_db/` : Dossier contenant la base vectorielle ChromaDB
- `requirements.txt` : Dépendances Python

## Exemple de flux

1. **Extraction** : Les documents sont extraits de HDFS par goal_id.
2. **Découpage** : Les documents sont découpés en chunks.
3. **Indexation** : Les chunks sont vectorisés et stockés dans ChromaDB.
4. **Recherche** : À partir d'une requête, on retrouve les passages les plus pertinents.

## Auteurs

- Saif (github.com/Saif1005) 