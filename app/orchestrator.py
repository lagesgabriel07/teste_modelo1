from app.audio_processing.audio_handler import AudioHandler
from app.audio_processing.transcriber import Transcriber
from rag.rag_pipeline import RAGPipeline

class Orchestrator:
    def __init__(self):
        """ Inicializa os módulos do pipeline completo """
        self.audio_handler = AudioHandler()  # Gerencia arquivos de áudio
        self.transcriber = Transcriber()  # Converte áudio para texto
        self.rag_pipeline = RAGPipeline()  # IA para responder perguntas

    def processar_ultimo_audio(self):
        """ Recupera o último áudio, transcreve e gera resposta """
        print("\n🎤 Buscando o último áudio salvo...")

        # Passo 1: Recuperar o último arquivo de áudio salvo
        caminho_audio = self.audio_handler.listar_arquivos_audio()
        if not caminho_audio:
            print("\n⚠️ Nenhum áudio encontrado!")
            return "Erro: Nenhum áudio disponível."

        print(f"\n🎧 Último áudio encontrado: {caminho_audio}")

        # Passo 2: Converter áudio em texto
        texto_transcrito = self.transcriber.transcrever_audio(str(caminho_audio))
        if not texto_transcrito:
            print("\n⚠️ Nenhuma transcrição gerada. Verifique o áudio.")
            return "Erro: Nenhuma transcrição foi feita."

        print(f"\n📝 Texto transcrito: {texto_transcrito}")

        # Passo 3: Enviar texto para o pipeline RAG
        resposta = self.rag_pipeline.responder_pergunta(texto_transcrito)

        print("\n💬 Resposta da IA:")
        print(resposta)

        return resposta

# Teste rápido
if __name__ == "__main__":
    orchestrator = Orchestrator()
    resposta_final = orchestrator.processar_ultimo_audio()
    print("\n✅ Processo concluído!")
