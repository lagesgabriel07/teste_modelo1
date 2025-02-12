from llama_cpp import Llama
from rag.retriever import KnowledgeBase

class LLAMAModel:
    def __init__(self, modelo_path="models/llama-7b.ggmlv3.q4_0.bin"):
        """ Inicializa o modelo LLAMA """
        self.model = Llama(model_path=modelo_path)
        self.kb = KnowledgeBase()
        self.kb.indexar_documentos()

    def gerar_resposta(self, pergunta):
        """ Busca contexto na base de conhecimento e gera resposta """
        documentos_relevantes = self.kb.buscar_conhecimento(pergunta)
        contexto = " ".join(documentos_relevantes[:3])  # Usa os 3 documentos mais relevantes

        prompt = f"Contexto:\n{contexto}\n\nPergunta: {pergunta}\n\nResposta:"
        resposta = self.model(prompt, max_tokens=200)
        return resposta['choices'][0]['text'].strip()

# Teste rápido
if __name__ == "__main__":
    llm = LLAMAModel()
    print(llm.gerar_resposta("Quais são os riscos psicossociais no trabalho?"))
