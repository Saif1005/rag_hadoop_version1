# API RAG avec Ollama

Ce projet implémente un système de Retrieval-Augmented Generation (RAG) avec FastAPI et Ollama.

## Prérequis

- Python 3.8+
- Ollama installé et configuré avec le modèle llama3.2:1b
- Un index FAISS et des données préalablement indexées

## Installation

1. Cloner le dépôt

```bash
git clone <repository_url>
cd <repository_directory>
```

2. Installer les dépendances

```bash
pip install -r requirements.txt
```

3. Installer le modèle spaCy français

```bash
python -m spacy download fr_core_news_sm
```

4. S'assurer qu'Ollama est installé et que le modèle est disponible

```bash
ollama pull llama3.2:1b
```

## Configuration

Vous pouvez configurer l'application en utilisant des variables d'environnement:

```bash
# Configuration Ollama
export OLLAMA_BASE_URL="http://localhost:11434"  # URL de l'API Ollama
export OLLAMA_MODEL="llama3.2:1b"  # Modèle à utiliser
export OLLAMA_TIMEOUT="30"  # Timeout en secondes
export USE_OLLAMA_CLIENT="false"  # Utiliser le client Python d'Ollama (true/false)

# Configuration des chemins
export FAISS_DB_PATH="faiss_db/index.index"  # Chemin vers l'index FAISS
export GOAL_DATA_DIR="faiss_db"  # Répertoire contenant les données des goals
```

## Exécution

### Démarrer le serveur

```bash
python app.py
```

Ou avec uvicorn directement:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Utilisation de l'API

### Documentation interactive

Accédez à la documentation interactive à l'adresse:

```
http://localhost:8000/docs
```

### Exemples de requêtes

#### Lister les goals disponibles

```bash
curl -X GET http://localhost:8000/goals
```

#### Rechercher des informations par goal_id

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "goal_id": "saifsaif1005",
    "query": "Qu\'est-ce que l\'intelligence artificielle?",
    "top_k": 5
  }'
```

#### Générer une réponse avec RAG

```bash
curl -X POST http://localhost:8000/query_and_generate \
  -H "Content-Type: application/json" \
  -d '{
    "goal_id": "saifsaif1005",
    "query": "Qu\'est-ce que l\'intelligence artificielle?",
    "top_k": 5,
    "use_rag": true
  }'
```

#### Chat avec contexte RAG

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "goal_id": "saifsaif1005",
    "query": "Qu\'est-ce que l\'intelligence artificielle?",
    "messages": [
      {
        "role": "user",
        "content": "Explique-moi l\'intelligence artificielle"
      }
    ],
    "model": "llama3.2:1b",
    "temperature": 0.7,
    "top_k": 5,
    "use_rag": true
  }'
```

#### Chat en streaming avec RAG (pour une meilleure UX)

```bash
curl -X POST http://localhost:8000/chat_stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "goal_id": "saifsaif1005",
    "query": "Qu\'est-ce que l\'intelligence artificielle?",
    "messages": [
      {
        "role": "user",
        "content": "Explique-moi l\'intelligence artificielle"
      }
    ],
    "model": "llama3.2:1b",
    "temperature": 0.7,
    "top_k": 5,
    "use_rag": true
  }'
```

#### Déboguer les problèmes d'Ollama

```bash
curl -X POST http://localhost:8000/debug/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Dis-moi bonjour"
      }
    ],
    "model": "llama3.2:1b"
  }'
```

## Intégration avec un front-end

Pour intégrer le streaming dans une application front-end, vous pouvez utiliser l'API EventSource:

```javascript
const eventSource = new EventSource('/chat_stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    goal_id: 'saifsaif1005',
    query: 'Explique-moi l\'intelligence artificielle',
    messages: [
      {
        role: 'user',
        content: 'Explique-moi l\'intelligence artificielle',
      },
    ],
  }),
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'sources') {
    // Afficher les sources
    console.log('Sources:', data.sources);
  } else if (data.type === 'content') {
    // Afficher le contenu généré
    console.log('Contenu:', data.content);
  } else if (data.type === 'done') {
    // Fermer la connexion
    eventSource.close();
  } else if (data.type === 'error') {
    // Gérer les erreurs
    console.error('Erreur:', data.error);
    eventSource.close();
  }
};

eventSource.onerror = (error) => {
  console.error('Erreur de connexion:', error);
  eventSource.close();
};
```

## Résolution des problèmes

### Format de réponse d'Ollama

Si vous rencontrez des erreurs liées au parsing des réponses d'Ollama, vérifiez :

1. La version d'Ollama (les versions récentes utilisent un streaming par défaut)
2. Les réponses brutes avec l'endpoint `/debug/ollama`
3. Essayez d'utiliser l'endpoint `/chat_stream` qui gère explicitement le streaming

### Problèmes de connexion à Ollama

- Vérifiez que Ollama est bien en cours d'exécution
- Vérifiez que le modèle spécifié est bien téléchargé (`ollama list`)
- Vérifiez l'URL d'accès à l'API Ollama

## Structure du projet

- `app.py`: Application FastAPI principale
- `retrivial_data_from_hdfs.py`: Fonctions de récupération des données depuis HDFS
- `query_by_goal_id.py`: Utilitaire de ligne de commande pour interroger par goal_id
- `faiss_db/`: Répertoire contenant l'index FAISS et les données des goals 