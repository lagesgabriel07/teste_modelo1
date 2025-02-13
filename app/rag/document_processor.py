import fitz  # PyMuPDF para manipulação de PDFs
import os
from pathlib import Path
from unicodedata import normalize

class DocumentProcessor:
    def __init__(self, pasta_documentos="data/documentos"):
        """ Inicializa o processador de documentos """
        self.pasta_documentos = Path(pasta_documentos)

    def listar_documentos(self):
        """ Lista e normaliza os documentos para evitar erros de codificação """
        if not self.pasta_documentos.exists():
            print(f"\n❌ Erro: A pasta '{self.pasta_documentos}' não existe!")
            return []

        arquivos = os.listdir(self.pasta_documentos)

        # Normaliza os nomes dos arquivos para evitar problemas com acentos
        arquivos_normalizados = [
            normalize("NFKD", arquivo).encode("ascii", "ignore").decode("utf-8")
            for arquivo in arquivos
        ]

        return [self.pasta_documentos / arquivo for arquivo in arquivos_normalizados]

    def extrair_texto_pdf(self, pdf_path):
        """ Extrai texto de um arquivo PDF usando PyMuPDF (fitz) """
        try:
            with fitz.open(pdf_path) as doc:
                texto = "\n".join([page.get_text("text") for page in doc])
            
            if texto.strip():
                return texto
            else:
                print(f"⚠️ Nenhum texto extraído do documento: {pdf_path} (pode ser um PDF de imagens).")
                return None
        except Exception as e:
            print(f"❌ Erro ao processar {pdf_path}: {e}")
            return None
