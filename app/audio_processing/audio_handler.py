from pathlib import Path

class AudioHandler:
    def __init__(self, pasta_audio: str = "data/audios"):
        self.pasta_audio = Path(pasta_audio)

    def listar_arquivos_audio(self):
        """ Lista os arquivos de áudio disponíveis e retorna o mais recente. """
        extensoes_validas = ('.mp3', '.wav', '.m4a', '.ogg', '.flac')

        if not self.pasta_audio.exists() or not self.pasta_audio.is_dir():
            print(f"Erro: O diretório '{self.pasta_audio}' não existe.")
            return None

        arquivos = [f for f in self.pasta_audio.iterdir() if f.suffix.lower() in extensoes_validas]
        arquivos.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        return arquivos[0] if arquivos else None
