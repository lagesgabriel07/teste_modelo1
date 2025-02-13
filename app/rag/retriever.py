import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from app.rag.document_processor import DocumentProcessor

class KnowledgeBase:
    def __init__(self):
        self.modelo_embedding = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.documentos = []
        self.processor = DocumentProcessor()

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
                vetor = self.modelo_embedding.encode([texto])[0]
                self.index.add(np.array([vetor], dtype=np.float32))

        print(f"\n✅ {len(self.documentos)} documentos foram indexados com sucesso no FAISS.")

    def buscar_conhecimento(self, consulta, top_k=3):
        """ Busca documentos mais relevantes para a consulta """
        
        if not self.documentos:
            print("\n❌ Erro: Nenhum documento foi indexado na base de conhecimento.")
            return []

        vetor_consulta = self.modelo_embedding.encode([consulta])[0]

        if self.index.ntotal == 0:
            print("\n❌ Erro: Nenhum documento foi armazenado no índice FAISS.")
            return []

        _, indices = self.index.search(np.array([vetor_consulta], dtype=np.float32), top_k)

        if len(indices[0]) == 0 or indices[0][0] == -1:
            print("\n📚 Nenhum documento relevante encontrado para a consulta.")
            return []

        resultados = [self.documentos[i][1] for i in indices[0] if i < len(self.documentos)]
        return resultados
