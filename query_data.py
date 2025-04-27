import os
import threading
import hashlib
import time
import concurrent.futures
import argparse
from dotenv import load_dotenv
from retrivial_data_from_hdfs import get_cached_context, retrieve_similar_chunks, retrieve_documents_by_goal_id
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

PROMPT_TEMPLATE = """
Répondez à la question en vous basant UNIQUEMENT sur le contexte suivant. Si le contexte ne contient pas d'information pertinente pour répondre à la question, répondez "Je ne trouve pas d'information pertinente dans le document pour répondre à cette question."

Contexte:
{context}

Question: {question}

Instructions: 
1. Réponse détaillée (3-5 phrases maximum)
2. Inclure toutes les aides mentionnées dans le contexte
3. Format simple sans caractères spéciaux
4. Pas de sauts de ligne
5. Ne pas inventer d'informations qui ne sont pas dans le contexte
6. Si vous n'êtes pas sûr, indiquez-le clairement
"""

# Cache pour les réponses générées
response_cache = {}
cache_lock = threading.Lock()

def get_cached_response(query_text: str, context_text: str) -> str:
    cache_key = hashlib.md5((query_text + context_text).encode()).hexdigest()
    with cache_lock:
        return response_cache.get(cache_key)

def cache_response(query_text: str, context_text: str, response: str):
    cache_key = hashlib.md5((query_text + context_text).encode()).hexdigest()
    with cache_lock:
        response_cache[cache_key] = response

def is_context_relevant(context_text: str, query_text: str) -> bool:
    if not context_text or len(context_text.strip()) < 10:
        return False
    query_words = set(query_text.lower().split())
    context_words = set(context_text.lower().split())
    common_words = query_words.intersection(context_words)
    return len(common_words) >= 1

def get_llm():
    # Utilise l'API OpenAI directement
    return OpenAI(api_key=OPENAI_API_KEY)

def query_rag(query_text: str, goal_id: str, similarity_threshold: float = 1, top_k: int = 10):
    start_time = time.time()
    if not query_text or query_text.strip() == "":
        return "La requête ne peut pas être vide."
    # Récupération du contexte
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(get_cached_context, query_text, goal_id, similarity_threshold, top_k)
        context_text = future.result()
    if not context_text or not is_context_relevant(context_text, query_text):
        return "Je ne trouve pas d'information pertinente dans le document pour répondre à cette question."
    cached_response = get_cached_response(query_text, context_text)
    if cached_response:
        print(f"Réponse récupérée du cache en {time.time() - start_time:.2f} secondes")
        return cached_response
    # Génération de la réponse avec OpenAI
    llm = get_llm()
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    response = llm.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    answer = response.choices[0].message.content.strip()
    cache_response(query_text, context_text, answer)
    print(f"Réponse générée en {time.time() - start_time:.2f} secondes")
    return answer

def debug_print_chunks(chunks):
    print("\n--- DEBUG : Chunks similaires trouvés ---")
    if not chunks:
        print("Aucun chunk similaire trouvé.")
        return
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk #{i}")
        print(f"  Score: {chunk.get('similarity', 0):.2f}")
        print(f"  Meta: {chunk.get('metadata', {})}")
        doc = chunk.get('document', '')
        print(f"  Texte: {doc[:200]}{'...' if len(doc) > 200 else ''}")
        print("-")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="La question à poser.")
    parser.add_argument("--goal_id", type=str, default=os.getenv("DEFAULT_GOAL_ID", "doc_1"), help="Le goal_id à interroger.")
    parser.add_argument("--top_k", type=int, default=10, help="Nombre de chunks à utiliser pour le contexte.")
    parser.add_argument("--similarity_threshold", type=float, default=0.2, help="Seuil de similarité pour les chunks.")
    parser.add_argument("--debug-chunks", action="store_true", help="Afficher les chunks similaires pour debug.")
    args = parser.parse_args()
    query_text = args.query_text
    goal_id = args.goal_id
    top_k = args.top_k
    similarity_threshold = args.similarity_threshold
    debug_chunks = args.debug_chunks
    print(f"Recherche de contexte pour goal_id={goal_id} ...")
    # Pour debug, afficher les chunks similaires trouvés
    if debug_chunks:
        chunks = retrieve_similar_chunks(goal_id, query_text, k=top_k, score_threshold=similarity_threshold)
        debug_print_chunks(chunks)
    context = get_cached_context(query_text, goal_id, similarity_threshold, top_k)
    print("Contexte extrait :\n", context)
    print("\nGénération de la réponse ...")
    response = query_rag(query_text, goal_id, similarity_threshold, top_k)
    print("\nRéponse générée :\n", response)

if __name__ == "__main__":
    main() 