import logging
import torch
from whisper import load_model
from pathlib import Path
from app.api.audio_processing.audio_handler import AudioHandler

# Configuração do logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, audio_handler: AudioHandler, model_name: str = "medium"):
        """
        Inicializa o transcritor com um manipulador de áudio e o modelo Whisper.
        """
        self.audio_handler = audio_handler
        self.model = self.load_whisper_model(model_name)

    def load_whisper_model(self, model_name: str):
        """
        Carrega o modelo Whisper, garantindo que esteja disponível para uso.
        """
        logger.info("Carregando o modelo Whisper: %s", model_name)
        try:
            return load_model(model_name)
        except Exception as e:
            logger.error("Erro ao carregar o modelo Whisper: %s", str(e))
            raise e

    def get_latest_audio_file(self) -> str:
        """
        Obtém o arquivo de áudio mais recente no diretório especificado.
        """
        audio_files = sorted(
            Path(self.audio_handler.audio_directory).glob("*.mp3"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        if not audio_files:
            logger.error("Nenhum arquivo de áudio encontrado no diretório: %s", self.audio_handler.audio_directory)
            raise FileNotFoundError("Nenhum arquivo de áudio encontrado.")
        
        latest_file = str(audio_files[0])
        logger.info("Último arquivo de áudio encontrado: %s", latest_file)
        return latest_file

    def transcribe_audio(self, filename: str = None) -> str:
        """
        Transcreve um arquivo de áudio. Se nenhum nome for fornecido, usa o mais recente.
        """
        if filename is None:
            filename = self.get_latest_audio_file()
        
        audio_path = self.audio_handler.get_audio_file(filename)
        logger.info("Transcrevendo áudio: %s", audio_path)
        
        try:
            result = self.model.transcribe(audio_path)
            transcription = result['text']
            logger.info("Transcrição concluída com sucesso.")
            return transcription
        except Exception as e:
            logger.error("Erro ao transcrever o áudio: %s", str(e))
            raise e

# Execução principal
if __name__ == "__main__":
    audio_directory = './data/audios'
    audio_handler = AudioHandler(audio_directory)
    transcriber = Transcriber(audio_handler)

    try:
        transcription = transcriber.transcribe_audio()  # Agora ele busca automaticamente o último arquivo
        print("Transcrição:", transcription)
    except Exception as e:
        print("Erro durante a transcrição:", str(e))
