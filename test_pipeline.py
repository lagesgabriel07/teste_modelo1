import sys
from pathlib import Path

# Define o diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent

# Adiciona os caminhos ao sys.path
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "app"))  # Agora "app" está corretamente referenciado

# Agora importe os módulos corretamente
from app.rag import KnowledgeBase
from app.rag import LLAMAModel
from app.audio_processing.transcriber import AudioTranscriber
from app.audio_processing.audio_handler import AudioHandler



def processar_audio():
    """ Transcreve um áudio da pasta 'data/audios/' """
    transcriber = AudioTranscriber(modelo='medium')
    caminho_audio = BASE_DIR / 'data' / 'audios' / 'audio_teste.mp3'  # Ajuste para o nome correto do seu arquivo
    
    if not caminho_audio.exists():
        print(f"\n❌ Erro: O arquivo de áudio '{caminho_audio}' não foi encontrado!")
        return None
    
    texto_transcrito = transcriber.transcrever_audio(caminho_audio)
    print('\n🔊 Transcrição:', texto_transcrito)
    return texto_transcrito

def buscar_conhecimento(texto_transcrito):
    """ Busca documentos relevantes no RAG """
    if not texto_transcrito:
        print("\n❌ Erro: Nenhuma transcrição disponível para buscar conhecimento.")
        return []

    kb = KnowledgeBase()
    kb.indexar_documentos()
    documentos = kb.buscar_conhecimento(texto_transcrito)

    if not documentos:
        print("\n📚 Nenhum documento relevante encontrado.")
        return []

    print('\n📚 Documentos encontrados:', documentos[:2])
    return documentos

def gerar_resposta(texto_transcrito, documentos):
    """ Usa o modelo LLAMA para gerar uma resposta baseada na busca no RAG """
    if not documentos:
        print("\n❌ Erro: Nenhum documento encontrado para gerar resposta.")
        return None

    llm = LLAMAModel()
    contexto = ' '.join(documentos[:3])  # Pega os três primeiros documentos
    resposta = llm.gerar_resposta(texto_transcrito, contexto)

    print('\n🤖 Resposta final:', resposta)
    return resposta


if __name__ == "__main__":
    print("\n🚀 Iniciando Teste do Pipeline...\n")

    # Executa o pipeline completo
    texto_transcrito = processar_audio()
    if texto_transcrito:
        documentos = buscar_conhecimento(texto_transcrito)
        gerar_resposta(texto_transcrito, documentos)

    print("\n✅ Teste finalizado com sucesso!")
