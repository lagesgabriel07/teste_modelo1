from llama_cpp import Llama

class LLAMAModel:
    def __init__(self, modelo_path="models/llama-7b.ggmlv3.q4_0.bin"):
        """ Carrega o modelo LLAMA """
        self.model = Llama(model_path=modelo_path)

    def gerar_resposta(self, pergunta, contexto):
        """ Gera resposta baseada na pergunta e no contexto fornecido. """
        prompt = f"Contexto:\n{contexto}\n\nPergunta: {pergunta}\n\nResposta:"
        resposta = self.model(prompt, max_tokens=200)
        return resposta['choices'][0]['text'].strip()

# Teste do LLAMA
if __name__ == "__main__":
    llm = LLAMAModel()
    contexto = "A inteligência artificial é um campo da ciência da computação focado no desenvolvimento de sistemas que podem realizar tarefas que normalmente exigiriam inteligência humana."
    print(llm.gerar_resposta("O que é IA?", contexto))
