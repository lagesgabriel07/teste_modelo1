from app.audio_processing.transcriber import AudioTranscriber
from app.audio_processing.audio_handler import AudioHandler

# Configuração do diretório
PASTA_AUDIOS = "C:\Users\Gabriel Lages\mentesegura\audios"

# Inicializar classes
audio_handler = AudioHandler(PASTA_AUDIOS)
transcriber = AudioTranscriber(modelo="medium")

# Obter o último arquivo de áudio e transcrever
ultimo_audio = audio_handler.listar_arquivos_audio()
if ultimo_audio:
    transcriber.transcrever_audio(ultimo_audio)
else:
    print("🚫 Nenhum arquivo de áudio encontrado!")
