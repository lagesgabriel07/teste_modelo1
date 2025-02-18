from rag.retriever import KnowledgeBase
from rag.llama_model import LLAMAModel

class RAGPipeline:
    def __init__(self):
        """ Inicializa o pipeline RAG (busca + IA) """
        self.retriever = KnowledgeBase()
        self.llama = LLAMAModel()

    def responder_pergunta(self, pergunta):
        """ Busca documentos relevantes e gera resposta """
        documentos_relevantes = self.retriever.buscar_conhecimento(pergunta)
        
        if not documentos_relevantes:
            return "⚠️ Nenhum documento relevante encontrado."

        resposta = self.llama.gerar_resposta(pergunta, documentos_relevantes)
        return resposta

# Teste rápido
if __name__ == "__main__":
    pipeline = RAGPipeline()
    print(pipeline.responder_pergunta("Quais são os sintomas da síndrome de Burnout?"))
