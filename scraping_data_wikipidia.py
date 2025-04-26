import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import subprocess
import os

def scrape_wikipedia_to_pdf_and_hdfs(
    url,
    local_pdf_path="wikipedia_scraping.pdf",
   
):
    try:
        # 1. Scraper la page
        print("🔎 Récupération de la page...")
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1').text.strip() if soup.find('h1') else "Sans titre"
        paragraphs = soup.find_all('p')
        content = "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])

        # 2. Générer le PDF (UTF-8 avec police Unicode)
        print("📝 Création du PDF...")
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        pdf.set_font("DejaVu", size=12)
        pdf.multi_cell(0, 10, title + "\n\n" + content)
        pdf.output(local_pdf_path)
        print(f"✅ PDF généré localement : {local_pdf_path}")

        # 3. Copier vers HDFS
     

    except Exception as e:
        print(f"❌ Une erreur s'est produite : {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    url = "https://fr.wikipedia.org/wiki/Intelligence_artificielle"
    scrape_wikipedia_to_pdf_and_hdfs(url)
