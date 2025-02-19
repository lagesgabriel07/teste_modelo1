import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

class LLAMAModel:
    def __init__(self, modelo="meta-llama/Llama-2-7b-chat-hf", modelo_embedding="all-MiniLM-L6-v2"):
        """ Inicializa o modelo LLaMA e o modelo de embeddings para RAG """

        # Carregamento otimizado com quantização (se disponível)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo = modelo
        self.pipeline = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(modelo, torch_dtype=torch.float16, device_map="auto"),
            tokenizer=AutoTokenizer.from_pretrained(modelo),
            max_new_tokens=500,  # Evita truncamentos errados
        )

        # Similaridade semântica (Sentence Transformers)
        self.embedding_model = SentenceTransformer(modelo_embedding)

    def selecionar_documentos_relevantes(self, pergunta, documentos):
        """ Seleciona os documentos mais relevantes  """

        if not documentos:
            return []

        # Calcula embeddings
        embeddings_docs = self.embedding_model.encode(documentos, convert_to_tensor=True)
        embedding_pergunta = self.embedding_model.encode(pergunta, convert_to_tensor=True)

        # Calcula similaridade
        scores = util.pytorch_cos_sim(embedding_pergunta, embeddings_docs)[0]
        indices_ordenados = scores.argsort(descending=True)

        # Retorna os 3 documentos mais relevantes
        return [documentos[i] for i in indices_ordenados[:3]]

    def gerar_resposta(self, pergunta, documentos_relevantes):
        """ Gera uma resposta baseada nos documentos mais relevantes usando LLaMA """

        if not documentos_relevantes:
            return "⚠️ Nenhum documento relevante encontrado para responder à pergunta."

        contexto = " ".join(documentos_relevantes)

        prompt = f"""Você é um assistente especializado. Responda à seguinte pergunta com base no contexto fornecido.

        Contexto:
        {contexto}

        Pergunta: {pergunta}

        Resposta:"""

        # Gera resposta com o modelo
        resposta = self.pipeline(prompt, max_new_tokens=500)

        texto_gerado = resposta[0]["generated_text"]

        # Usa regex para extrair corretamente a resposta
        resposta_final = re.search(r"Resposta:\s*(.*)", texto_gerado, re.DOTALL)
        return resposta_final.group(1).strip() if resposta_final else texto_gerado.strip()


# Teste rápido
if __name__ == "__main__":
    modelo_llama = LLAMAModel()
    documentos = [
        "O Burnout é uma condição psicológica causada por estresse crônico no trabalho.",
        "A síndrome de Burnout apresenta sintomas como cansaço extremo, falta de motivação e irritabilidade."
    ]
    
    documentos_relevantes = modelo_llama.selecionar_documentos_relevantes("Quais são os sintomas da síndrome de Burnout?", documentos)
    print(modelo_llama.gerar_resposta("Quais são os sintomas da síndrome de Burnout?", documentos_relevantes))
