import os
import PyPDF2
import tiktoken
import openai
from dotenv import load_dotenv
import pyhdfs  # Assurez-vous que pyhdfs est installé et configuré correctement

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def lire_pdf(file_paths):
    textes = {}
    for file_path in file_paths:
        texte = ""
        with open(file_path, "rb") as f:
            lecteur = PyPDF2.PdfReader(f)
            for page in lecteur.pages:
                texte += page.extract_text()
        textes[file_path] = texte
    return textes

def split_text_in_chunks(text, chunk_token_size=1000, overlap=300, model_name="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model_name)
    tokens = tokenizer.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_token_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_token_size - overlap
    return chunks

def create_hdfs_directory(hdfs_client, goal_id):
    directory_path = f"/user/hadoop/goals/{goal_id}"
    if not hdfs_client.exists(directory_path):
        hdfs_client.mkdirs(directory_path)
    print(f"Répertoire HDFS créé (ou déjà existant) à : {directory_path}")

def sauvegarder_chunks_dans_hdfs(chunks, goal_id, hdfs_client):
    create_hdfs_directory(hdfs_client, goal_id)
    output_hdfs_path = f"/user/hadoop/goals/{goal_id}/output_chunks.txt"
    if hdfs_client.exists(output_hdfs_path):
        hdfs_client.delete(output_hdfs_path)
        print(f"Le fichier existant {output_hdfs_path} a été supprimé.")
    data_to_write = "\n".join(chunks)
    hdfs_client.create(output_hdfs_path, data=data_to_write)
    print(f"Chunks sauvegardés dans HDFS à l'emplacement : {output_hdfs_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    file_paths = [
        "/mnt/c/Users/saifa/projet_rag_hadoop/wikipedia_scraping.pdf",
        "/mnt/c/Users/saifa/projet_rag_hadoop/ai2.pdf"  # Ajoutez d'autres fichiers PDF ici
    ]
    
    # Connexion à HDFS via pyhdfs
    hdfs_client = pyhdfs.HdfsClient(hosts='localhost:50070')  # Utilise le port WebHDFS (50070)
    
    # Traiter chaque fichier avec un goal_id unique
    for i, file_path in enumerate(file_paths):
        goal_id = f"doc_{i + 1}"  # Exemple de goal_id unique
        texte_complet = lire_pdf([file_path])[file_path]
        chunks = split_text_in_chunks(texte_complet)
        sauvegarder_chunks_dans_hdfs(chunks, goal_id, hdfs_client)
        print(f"{len(chunks)} chunks sauvegardés dans HDFS sous /user/hadoop/goals/{goal_id}/output_chunks.txt")
