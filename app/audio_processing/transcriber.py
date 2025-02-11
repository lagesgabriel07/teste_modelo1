import whisper
from pathlib import Path

# Definir a pasta de áudio
PASTA_AUDIOS = Path(r"C:\Users\Gabriel Lages\mentesegura\audios")
PASTA_TRANSCRICOES = PASTA_AUDIOS / "transcricoes"
PASTA_TRANSCRICOES.mkdir(exist_ok=True)  # Criar pasta caso não exista

class AudioTranscriber:
    def __init__(self, modelo="medium"):
        """ Inicializa o modelo Whisper. """
        self.model = whisper.load_model(modelo)

    def transcrever_audio(self, arquivo_audio: Path):
        """ Transcreve um arquivo de áudio usando Whisper. """
        if not arquivo_audio.exists():
            raise FileNotFoundError(f"O arquivo '{arquivo_audio}' não foi encontrado!")

        print(f"\n🔍 Processando: {arquivo_audio}")
        resultado = self.model.transcribe(str(arquivo_audio), language="pt")
        texto_transcricao = resultado["text"]
        print("\n📝 Transcrição:\n", texto_transcricao)

        # Salvar transcrição
        caminho_saida = PASTA_TRANSCRICOES / f"{arquivo_audio.stem}_transcricao.txt"
        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(texto_transcricao)

        print(f"\n✅ Transcrição salva em: {caminho_saida}")
        return texto_transcricao
