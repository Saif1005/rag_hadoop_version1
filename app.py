"""
API FastAPI pour la récupération de données et la génération avec Ollama
"""

import os
import json
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

from retrivial_data_from_hdfs import (
    list_available_goals,
    enhanced_similarity_search,

    GOAL_DATA_DIR
)

# Configuration de l'API Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
USE_OLLAMA_CLIENT = os.getenv("USE_OLLAMA_CLIENT", "false").lower() in ["true", "1", "yes"]

# Création de l'application FastAPI
app = FastAPI(
    title="API RAG avec Ollama",
    description="API pour la récupération de données et la génération avec Ollama",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles de données
class QueryRequest(BaseModel):
    goal_id: str
    query: str
    top_k: int = 5
    min_score: float = 0.05
    use_rag: bool = True
    
class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    goal_id: Optional[str] = None
    query: Optional[str] = None
    messages: List[ChatMessage]
    model: str = OLLAMA_MODEL
    temperature: float = 0.7
    top_k: int = 5
    use_rag: bool = True
    
class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []

# Routes API
@app.get("/")
async def read_root():
    """Route racine"""
    return {"message": "API RAG avec Ollama", "version": "1.0.0"}

@app.get("/goals")
async def get_goals():
    """Récupérer la liste des goals disponibles"""
    goals = list_available_goals()
    return {"goals": goals}

@app.post("/search")
async def search(request: QueryRequest):
    try:
        results = enhanced_similarity_search(
            query_text=request.query,
            goal_id=request.goal_id,
            top_k=request.top_k,
            min_score=0.05,
            max_distance=2.0
        )

        if not results:
            return {
                "status": "no_results",
                "suggestions": [
                    "Augmenter max_distance (actuel: 1.5)",
                    "Réduire min_score (actuel: {request.min_score})"
                ]
            }

        return {
            "status": "success",
            "count": len(results),
            "results": results
        }

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

async def generate_with_ollama_http(messages: List[ChatMessage], model: str = OLLAMA_MODEL, temperature: float = 0.7):
    """Générer une réponse avec Ollama via HTTP"""
    try:
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": model,
                    "messages": formatted_messages,
                    "options": {
                        "temperature": temperature
                    }
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Erreur Ollama: {response.text}"
                )
            
            # Vérifier si la réponse est du JSON valide ou un flux de réponses
            try:
                # Essayer de parser comme du JSON simple
                result = response.json()
                
                if "message" in result:
                    return result["message"]["content"]
                else:
                    # Fallback si le format ne correspond pas
                    return response.text
                    
            except json.JSONDecodeError:
                # Si ce n'est pas un JSON simple, c'est peut-être un flux de réponses
                # Chaque ligne étant un objet JSON
                full_text = ""
                try:
                    lines = response.text.strip().split("\n")
                    for line in lines:
                        try:
                            # Parser chaque ligne comme un objet JSON
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                full_text += chunk["message"]["content"]
                        except json.JSONDecodeError:
                            # Ignorer les lignes qui ne sont pas du JSON valide
                            continue
                    
                    return full_text if full_text else response.text
                except Exception as e:
                    # En cas d'erreur, retourner le texte brut
                    print(f"Erreur lors du parsing du flux: {str(e)}")
                    return response.text
    
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Erreur de connexion à Ollama: {str(e)}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération: {str(e)}"
        )

async def generate_with_ollama_client(messages: List[ChatMessage], model: str = OLLAMA_MODEL, temperature: float = 0.7):
    """Générer une réponse avec le client Python Ollama"""
    try:
        import ollama
        
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        try:
            response = ollama.chat(
                model=model,
                messages=formatted_messages,
                options={"temperature": temperature}
            )
            
            if isinstance(response, dict) and "message" in response:
                return response["message"]["content"]
            elif isinstance(response, str):
                return response
            else:
                # Gérer divers types de réponses
                return str(response)
        except Exception as e:
            # Fallback au HTTP si le client Python échoue
            print(f"Erreur avec client Ollama, fallback à HTTP: {str(e)}")
            return await generate_with_ollama_http(messages, model, temperature)
    
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Client Python Ollama non installé. Utilisez 'pip install ollama'."
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération: {str(e)}"
        )

async def generate_with_ollama(messages: List[ChatMessage], model: str = OLLAMA_MODEL, temperature: float = 0.7):
    """Fonction principale pour générer une réponse avec Ollama"""
    if USE_OLLAMA_CLIENT:
        return await generate_with_ollama_client(messages, model, temperature)
    else:
        return await generate_with_ollama_http(messages, model, temperature)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Discuter avec le modèle avec RAG intégré"""
    try:
        context_texts = []
        sources = []
        
        # Si RAG est activé et qu'il y a un goal_id et une requête
        if request.use_rag and request.goal_id and request.query:
            # Récupérer les informations pertinentes
            results, indices, distances = similarity_search_by_goal_id(
                query_text=request.query,
                goal_id=request.goal_id,
                top_k=request.top_k,
                max_distance=0.8
            )
            
            if not results:
                raise HTTPException(
                    status_code=404,
                    detail="Aucun contexte pertinent trouvé (distance > 0.8)"
                )
            
            # Préparer le contexte pour le modèle
            if results:
                context_texts = results
                
                # Préparer les sources pour la réponse
                for i, (result, dist) in enumerate(zip(results, distances)):
                    sources.append({
                        "content": result[:100] + "..." if len(result) > 100 else result,
                        "distance": float(dist),
                        "index": int(indices[i]) if i < len(indices) else i
                    })
        
        # Préparer les messages avec le contexte si disponible
        messages = list(request.messages)
        
        if context_texts and request.use_rag:
            # Transformer la liste de contextes en une seule chaîne
            context_str = "\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(context_texts)])
            
            # Ajouter un message système avec le contexte
            system_message = ChatMessage(
                role="system",
                content=(
                    "Tu es un assistant IA expert. Utilise les passages suivants pour répondre à la question de l'utilisateur. "
                    "Si tu ne trouves pas la réponse dans les passages, dis-le honnêtement.\n\n"
                    f"{context_str}"
                )
            )
            
            # Insérer le message système au début
            messages.insert(0, system_message)
        
        # Générer la réponse
        answer = await generate_with_ollama(
            messages=messages,
            model=request.model,
            temperature=request.temperature
        )
        
        return ChatResponse(answer=answer, sources=sources)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/query_and_generate")
async def query_and_generate(request: QueryRequest):
    """Rechercher des informations et générer une réponse en une seule étape"""
    try:
        # Récupérer les informations pertinentes
        if request.use_rag:
            results, indices, distances = similarity_search_by_goal_id(
                query_text=request.query,
                goal_id=request.goal_id,
                top_k=request.top_k,
                max_distance=0.8
            )
            
            if not results:
                raise HTTPException(
                    status_code=404,
                    detail="Aucun contexte pertinent trouvé (distance > 0.8)"
                )
            
            # Formatter les sources pour la réponse
            sources = []
            for i, (result, dist) in enumerate(zip(results, distances)):
                sources.append({
                    "content": result[:100] + "..." if len(result) > 100 else result,
                    "distance": float(dist),
                    "index": int(indices[i]) if i < len(indices) else i
                })
                
            # Préparer le contexte pour le modèle
            if results:
                context_str = "\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(results)])
                
                # Créer les messages pour la génération
                messages = [
                    ChatMessage(
                        role="system",
                        content=(
                            "Tu es un assistant IA expert. Utilise les passages suivants pour répondre à la question de l'utilisateur. "
                            "Si tu ne trouves pas la réponse dans les passages, dis-le honnêtement.\n\n"
                            f"{context_str}"
                        )
                    ),
                    ChatMessage(
                        role="user",
                        content=request.query
                    )
                ]
        else:
            # Sans RAG, juste envoyer la requête au modèle
            messages = [
                ChatMessage(
                    role="user",
                    content=request.query
                )
            ]
            sources = []
        
        # Générer la réponse
        answer = await generate_with_ollama(messages=messages)
        
        return ChatResponse(answer=answer, sources=sources)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/debug/ollama")
async def debug_ollama(request: ChatRequest):
    """
    Point de terminaison pour déboguer les communications avec Ollama.
    Retourne la réponse brute d'Ollama.
    """
    try:
        # Formater les messages
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Appeler directement l'API Ollama
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": request.model,
                    "messages": formatted_messages,
                    "options": {
                        "temperature": request.temperature
                    }
                }
            )
            
            # Retourner la réponse brute et les informations de débogage
            content_type = response.headers.get("content-type", "")
            
            return {
                "status_code": response.status_code,
                "content_type": content_type,
                "raw_response": response.text,
                "is_json": "application/json" in content_type,
                "parsed_json": response.json() if "application/json" in content_type else None,
                "headers": dict(response.headers),
                "request_sent": {
                    "model": request.model,
                    "messages": formatted_messages,
                    "options": {
                        "temperature": request.temperature
                    }
                }
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du débogage de l'API Ollama: {str(e)}"
        )

@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """
    Discuter avec le modèle avec RAG intégré et streaming de la réponse.
    Ne génère pas de réponse si la distance > 0.8.
    """
    try:
        context_texts = []
        sources = []
        
        # Si RAG est activé et qu'il y a un goal_id et une requête
        if request.use_rag and request.goal_id and request.query:
            results, indices, distances = similarity_search_by_goal_id(
                query_text=request.query,
                goal_id=request.goal_id,
                top_k=request.top_k,
                max_distance=0.8  # Ajout du seuil
            )
            
            if not results:  # Aucun résultat valide trouvé
                async def no_results():
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Aucun contexte pertinent trouvé (distance > 0.8)'})}\n\n"
                    yield "event: end\ndata: {}\n\n"
                
                return StreamingResponse(no_results(), media_type="text/event-stream")
            
            # Préparer le contexte
            context_texts = results
            sources = [
                {
                    "content": result[:100] + "..." if len(result) > 100 else result,
                    "distance": float(dist),
                    "index": int(indices[i]) if i < len(indices) else i
                }
                for i, (result, dist) in enumerate(zip(results, distances))
            ]
        
        messages = list(request.messages)
        
        if context_texts and request.use_rag:
            context_str = "\n\n".join([f"Passage {i+1}:\n{text}" for i, text in enumerate(context_texts)])
            system_message = ChatMessage(
                role="system",
                content=(
                    "Tu es un assistant IA expert. Utilise les passages suivants pour répondre à la question de l'utilisateur. "
                    "Si tu ne trouves pas la réponse dans les passages, dis-le honnêtement.\n\n"
                    f"{context_str}"
                )
            )
            messages.insert(0, system_message)
        
        async def generate_stream():
            # Envoyer les sources d'abord
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            # Préparer la requête pour Ollama
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # Appeler Ollama en mode streaming
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": request.model,
                        "messages": formatted_messages,
                        "options": {"temperature": request.temperature},
                        "stream": True
                    }
                ) as response:  # Définition de la variable response ici
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'type': 'error', 'error': f'Erreur Ollama: {response.status_code}'})}\n\n"
                        return
                    
                    buffer = ""
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                content = chunk["message"]["content"]
                                # Nettoyer et bufferiser
                                content = content.replace(" .", ".").replace(" ,", ",")
                                buffer += content
                                # Envoyer quand une phrase est complète
                                if buffer.endswith(('.', '!', '?', '\n')):
                                    yield f"data: {json.dumps({'type': 'content', 'content': buffer.strip()})}\n\n"
                                    buffer = ""
                        except json.JSONDecodeError:
                            continue
                    
                    # Envoyer le reste du buffer
                    if buffer:
                        yield f"data: {json.dumps({'type': 'content', 'content': buffer.strip()})}\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/event-stream") 
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

# Démarrage de l'application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
