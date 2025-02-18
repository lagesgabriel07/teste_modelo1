import fitz  # PyMuPDF para manipulação de PDFs
import os
from pathlib import Path

class DocumentProcessor:
    def __init__(self, pasta_documentos="data/documentos"):
        """ Inicializa o processador de documentos """
        self.pasta_documentos = Path(pasta_documentos)

    def listar_documentos(self):
        """ Lista os documentos na pasta sem modificar os nomes """
        if not self.pasta_documentos.exists():
            print(f"\n❌ Erro: A pasta '{self.pasta_documentos}' não existe!")
            return []

        return [arquivo for arquivo in self.pasta_documentos.iterdir() if arquivo.is_file()]

    def extrair_texto_pdf(self, pdf_path):
        """ Extrai texto de um arquivo PDF usando PyMuPDF (fitz) """
        try:
            with fitz.open(pdf_path) as doc:
                texto = "\n".join([page.get_text("blocks") for page in doc])  # Melhor para estruturar o texto

            if texto.strip():
                return texto
            else:
                print(f"⚠️ Nenhum texto extraído do documento: {pdf_path} (pode ser um PDF de imagens).")
                return None
        except Exception as e:
            print(f"❌ Erro ao processar {pdf_path}: {e}")
            return None
