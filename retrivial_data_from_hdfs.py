import os
import hashlib
import shutil
import concurrent.futures
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from chromadb import PersistentClient as Chroma
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.schema import Document
import chromadb
import pyhdfs
import sys
import contextlib

load_dotenv()
CHROMA_PATH = os.getenv('CHROMA_DB_PATH', 'chroma_db')

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks: list[Document], goal_id: str):
    for i, chunk in enumerate(chunks):
        chunk.metadata["goal_id"] = goal_id
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "0")
        filename = os.path.basename(source)
        snippet = chunk.page_content[:50] if chunk.page_content else ""
        unique_string = f"{filename}-{page}-{i}-{snippet}"
        chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()[:6]
        chunk.metadata["chunk_id"] = chunk_id
    return chunks

def get_embedding_function():
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def add_to_chroma(chunks: list[Document], goal_id: str, USE_CHROMA: bool = True):
    if not USE_CHROMA:
        print("chroma n'est pas activé")
        return {"messsage" : "chroma n'est pas activé"}
    try:
        if not chunks:
            print("⚠️ Aucun document à ajouter à Chroma.")
            return {"message": "Aucun document à ajouter à Chroma."}
        embedding_function = get_embedding_function()
        os.makedirs(CHROMA_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_or_create_collection(goal_id)
        # Générer les embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_function.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        ids = [chunk.metadata.get("chunk_id") for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    ids=ids,
                    metadatas=metadatas
                )
        return {"message": f"{len(chunks)} new chunks added to Chroma successfully!"}
    except Exception as e:
        print(f"❌ Error in add_to_chroma: {str(e)}")
        raise Exception(f"Internal server error: {str(e)}")

def goal_id_exists(goal_id: str , USE_CHROMA: bool = True) -> bool:
    if not USE_CHROMA:
        return False 
    try:
        if os.path.exists(CHROMA_PATH):
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collections = client.list_collections()
            return goal_id in collections
        return False
    except Exception as e:
        print(f"Erreur lors de la vérification de l'existence du goal_id: {str(e)}")
        return False

def retrieve_documents_by_goal_id(goal_id: str , USE_CHROMA: bool = True) -> list:
    if not USE_CHROMA: 
        return []
    try:
        if os.path.exists(CHROMA_PATH):
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            collections = client.list_collections()
            if goal_id not in collections:
                return []
            collection = client.get_collection(goal_id)
            results = collection.get()
            documents = []
            if results and "documents" in results and len(results["documents"]) > 0:
                for i, doc_text in enumerate(results["documents"]):
                    doc_id = results["ids"][i] if "ids" in results and i < len(results["ids"]) else f"chroma_{i}"
                    metadata = results["metadatas"][i] if "metadatas" in results and i < len(results["metadatas"]) else {}
                    documents.append({
                        "id": doc_id,
                        "metadata": metadata,
                        "document": doc_text,
                    })
            return documents
        else:
            return []
    except Exception as e:
        print(f"Erreur lors de la récupération des documents: {str(e)}")
        return []

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

def get_cached_context(query_text: str, goal_id: str, similarity_threshold: float = 1, top_k: int = 10 , USE_CHROMA:bool = True) -> str:
    if not USE_CHROMA: 
        return ""
    try:
        similar_chunks = retrieve_similar_chunks(
            goal_id=goal_id,
            query_text=query_text,
            k=top_k,
            score_threshold=similarity_threshold
        )
        if not similar_chunks:
            return ""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_chunks = list(executor.map(lambda chunk: chunk["document"][:200] + ("..." if len(chunk["document"]) > 200 else ""), similar_chunks))
        context_parts = [chunk for chunk in processed_chunks if chunk]
        total_context = " ".join(context_parts)
        if len(total_context) > 3000:
            return total_context[:3000] + "..."
        return total_context
    except Exception as e:
        print(f"Erreur lors de la récupération du contexte: {str(e)}")
        return ""

def get_chunks_from_hdfs(goal_id, hdfs_host='localhost', hdfs_port=50070):
    hdfs_client = pyhdfs.HdfsClient(hosts=f'{hdfs_host}:{hdfs_port}')
    hdfs_path = f"/user/hadoop/goals/{goal_id}/output_chunks.txt"
    if not hdfs_client.exists(hdfs_path):
        print(f"Fichier {hdfs_path} non trouvé dans HDFS.")
        return []
    data = hdfs_client.open(hdfs_path).read().decode('utf-8')
    chunks = data.split('\n')
    return chunks

def index_hdfs_chunks_to_chroma(goal_id, hdfs_host='localhost', hdfs_port=50070):
    # Vérifier si déjà indexé
    if goal_id_exists(goal_id):
        print(f"⚠️ Le goal_id '{goal_id}' existe déjà dans Chroma. Indexation ignorée.")
        return
    chunks = get_chunks_from_hdfs(goal_id, hdfs_host, hdfs_port)
    if not chunks:
        print(f"Aucun chunk trouvé pour {goal_id}")
        return
    documents = [Document(page_content=chunk, metadata={"source": f"hdfs:{goal_id}"}) for chunk in chunks if chunk.strip()]
    documents = calculate_chunk_ids(documents, goal_id)
    result = add_to_chroma(documents, goal_id)
    print(result)

def main():
    # Lister tous les goal_id présents dans HDFS
    hdfs_client = pyhdfs.HdfsClient(hosts='localhost:50070')
    base_path = "/user/hadoop/goals/"
    if not hdfs_client.exists(base_path):
        print("Aucun dossier goals/ trouvé dans HDFS.")
        return
    goal_ids = hdfs_client.listdir(base_path)
    for goal_id in goal_ids:
        print(f"Indexation automatique pour {goal_id}")
        index_hdfs_chunks_to_chroma(goal_id, hdfs_host='localhost', hdfs_port=50070)

if __name__ == "__main__":
    main()
    # Debug : afficher les 10 premiers chunks d'un goal_id
    goal_id = "saifsaif1005"
    docs = retrieve_documents_by_goal_id(goal_id)
    print(f"\nNombre de chunks pour {goal_id} : {len(docs)}")
    for i, d in enumerate(docs[:10], 1):
        print(f"\nChunk #{i}")
        print(f"Meta: {d['metadata']}")
        print(f"Texte: {d['document'][:300]}{'...' if len(d['document']) > 300 else ''}")
 

