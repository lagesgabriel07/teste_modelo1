import faiss
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from app.rag.document_processor import DocumentProcessor

class KnowledgeBase:
    def __init__(self, index_path="./models/knowledge_base.faiss"):
        """ Inicializa a base de conhecimento FAISS e carrega o modelo de embeddings """
        self.modelo_embedding = SentenceTransformer("all-MiniLM-L6-v2")
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(384)  # Vetor de tamanho 384 (MiniLM-L6-v2)
        self.documentos = []
        self.processor = DocumentProcessor()

        # Tenta carregar índice salvo
        self._carregar_indice()

    def _salvar_indice(self):
        """ Salva o índice FAISS no disco """
        if self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_path)
            print(f"\n💾 Índice FAISS salvo em {self.index_path}")

    def _carregar_indice(self):
        """ Carrega o índice FAISS salvo anteriormente """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"\n📂 Índice FAISS carregado de {self.index_path}")

    def indexar_documentos(self):
        """ Carrega e indexa os documentos da pasta usando embeddings """
        arquivos = self.processor.listar_documentos()

        if not arquivos:
            print("\n📂 Nenhum documento encontrado para indexação.")
            return

        for arquivo in arquivos:
            texto = self.processor.extrair_texto_pdf(arquivo)
            if texto:
                self.documentos.append((arquivo.name, texto))
                vetor = self.modelo_embedding.encode(texto)  
                self.index.add(np.array([vetor], dtype=np.float32))

        self._salvar_indice()
        print(f"\n✅ {len(self.documentos)} documentos foram indexados com sucesso no FAISS.")
