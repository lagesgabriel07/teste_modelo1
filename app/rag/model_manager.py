from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class ModelManager:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf", cache_dir="./models/llama-2"):
        """ Inicializa e carrega o modelo LLaMA, garantindo que o diretório de cache exista """

        # 🔹 Garante que o diretório de cache do modelo existe
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"📂 Criado diretório para armazenar modelo: {cache_dir}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 🔹 Carrega tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        # 🔹 Carrega modelo LLaMA
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto"
        )

        print(f"✅ Modelo {model_name} carregado e armazenado em {cache_dir}!")

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
