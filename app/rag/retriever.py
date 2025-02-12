import os
import fitz  # PyMuPDF para leitura de PDFs
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    def __init__(self, pasta_dados="data/documentos", modelo_embedding="sentence-transformers/all-MiniLM-L6-v2"):
        """ Inicializa a base de conhecimento """
        self.pasta_dados = pasta_dados
        self.modelo_embedding = SentenceTransformer(modelo_embedding)
        self.index = None
        self.docs = []

    def carregar_documentos(self):
        """ Lê os arquivos de texto e PDF dentro da pasta e armazena os conteúdos. """
        arquivos = [os.path.join(self.pasta_dados, f) for f in os.listdir(self.pasta_dados) if f.endswith(('.txt', '.pdf'))]

        for arquivo in arquivos:
            if arquivo.endswith('.txt'):
                with open(arquivo, 'r', encoding='utf-8') as f:
                    self.docs.append(f.read())

            elif arquivo.endswith('.pdf'):
                texto_pdf = self.extrair_texto_pdf(arquivo)
                if texto_pdf.strip():
                    self.docs.append(texto_pdf)

        print(f"✅ {len(self.docs)} documentos carregados.")

    def extrair_texto_pdf(self, caminho_pdf):
        """ Extrai texto de um arquivo PDF usando PyMuPDF (fitz). """
        texto = ""
        try:
            with fitz.open(caminho_pdf) as doc:
                for pagina in doc:
                    texto += pagina.get_text("text") + "\n"
        except Exception as e:
            print(f"❌ Erro ao ler PDF {caminho_pdf}: {e}")
        return texto

    def indexar_documentos(self):
        """ Gera embeddings dos documentos e cria um índice FAISS para busca. """
        if not self.docs:
            self.carregar_documentos()

        embeddings = self.modelo_embedding.encode(self.docs, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        print("🔍 Indexação completa!")

    def buscar_conhecimento(self, consulta, top_k=3):
        """ Busca informações relevantes nos documentos indexados. """
        if self.index is None:
            print("⚠️ Nenhum documento indexado! Execute `indexar_documentos()` primeiro.")
            return []

        embedding_consulta = self.modelo_embedding.encode([consulta], convert_to_numpy=True)
        distancias, indices = self.index.search(embedding_consulta, top_k)
        resultados = [self.docs[i] for i in indices[0]]
        return resultados

# Testando a indexação e busca
if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.indexar_documentos()
    print(kb.buscar_conhecimento("O que é psicodinâmica do trabalho?"))
