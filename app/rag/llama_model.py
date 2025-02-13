from transformers import pipeline

class LLAMAModel:
    def __init__(self):
        """ Inicializa o modelo de linguagem """
        self.pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

    def gerar_resposta(self, pergunta, documentos_relevantes):
        """ Gera uma resposta usando LLAMA com base nos documentos encontrados """

        if not documentos_relevantes:
            return "⚠️ Nenhum documento relevante encontrado para responder à pergunta."

        # Concatena os documentos mais relevantes para formar o contexto
        contexto = " ".join(documentos_relevantes[:3])

        # Formata o prompt de entrada
        prompt = f"""Você é um assistente especializado. Responda a seguinte pergunta com base no contexto fornecido.

        Contexto:
        {contexto}

        Pergunta: {pergunta}

        Resposta:"""

        # Gera resposta com o modelo LLaMA
        resposta = self.pipeline(prompt, max_length=300, truncation=True)

        return resposta[0]["generated_text"].split("Resposta:")[-1].strip()

# Teste rápido
if __name__ == "__main__":
    modelo_llama = LLAMAModel()
    documentos = ["O Burnout é uma condição psicológica causada por estresse crônico no trabalho."]
    print(modelo_llama.gerar_resposta("Quais são os sintomas da síndrome de Burnout?", documentos))
