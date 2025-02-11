import openai
import json
from transformers import RagTokenizer, RagSequenceForGeneration
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Configuração da API da OpenAI
openai.api_key = "sua_chave_api_aqui"  # Substitua pela sua chave da OpenAI

# 2. Função para transcrever áudio
def transcrever_audio(caminho_audio):
    with open(caminho_audio, "rb") as arquivo_audio:
        transcricao = openai.Audio.transcribe("whisper-1", arquivo_audio)
    return transcricao["text"]

# 3. Função para carregar a base de dados
def carregar_base_de_dados(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        dados = json.load(f)
    return dados

# 4. Função para criar índice FAISS
def criar_indice_faiss(dados):
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Modelo leve para embeddings
    textos = [item["text"] for item in dados]
    embeddings = model.encode(textos, convert_to_tensor=True).cpu().numpy()

    # Criar índice FAISS
    dimensao = embeddings.shape[1]
    indice = faiss.IndexFlatL2(dimensao)  # Índice de busca por similaridade
    indice.add(embeddings)
    return indice, textos

# 5. Função para buscar documentos relevantes
def buscar_documentos_relevantes(consulta, indice, textos, top_k=2):
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embedding_consulta = model.encode(consulta, convert_to_tensor=True).cpu().numpy()
    embedding_consulta = np.array([embedding_consulta])  # Formatar para FAISS

    # Buscar os documentos mais próximos
    distancias, indices = indice.search(embedding_consulta, top_k)
    documentos_relevantes = [textos[i] for i in indices[0]]
    return documentos_relevantes

# 6. Função para gerar respostas com RAG
def gerar_resposta_rag(consulta, dados, indice, textos):
    # Buscar documentos relevantes
    documentos_relevantes = buscar_documentos_relevantes(consulta, indice, textos)

    # Concatenar documentos para o modelo RAG
    contexto = " ".join(documentos_relevantes)

    # Carregar o modelo RAG
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

    # Preparar a entrada para o modelo
    inputs = tokenizer(consulta, contexto, return_tensors="pt")

    # Gerar a resposta
    outputs = model.generate(input_ids=inputs["input_ids"])
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta

# 7. Função principal
if __name__ == "__main__":
    # Transcrever o áudio
    caminho_audio = "audios/audio_teste.mp3"  # Substitua pelo caminho do seu áudio
    texto_transcrito = transcrever_audio(caminho_audio)
    print("Texto transcrito:", texto_transcrito)

    # Carregar base de dados
    caminho_arquivo = "dados/base_de_dados.json"
    dados = carregar_base_de_dados(caminho_arquivo)

    # Criar índice FAISS
    indice, textos = criar_indice_faiss(dados)

    # Gerar resposta com RAG
    resposta = gerar_resposta_rag(texto_transcrito, dados, indice, textos)
    print("Resposta:", resposta)