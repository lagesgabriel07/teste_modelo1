import whisper
from pathlib import Path

# Definir os caminhos corretamente dentro da pasta `data`
PASTA_AUDIOS = Path("data/audios")  # Agora busca dentro da pasta do projeto
PASTA_TRANSCRICOES = Path("data/transcricoes")  

# Criar a pasta de transcrições caso não exista
PASTA_TRANSCRICOES.mkdir(parents=True, exist_ok=True)

class AudioTranscriber:
    def __init__(self, modelo="medium"):
        """ Inicializa o modelo Whisper. """
        self.model = whisper.load_model(modelo)

    def transcrever_audio(self, arquivo_audio: Path):
        """ Transcreve um arquivo de áudio usando Whisper. """

        if not arquivo_audio.exists():
            raise FileNotFoundError(f"O arquivo '{arquivo_audio}' não foi encontrado!")

        print(f"\n🔊 Processando: {arquivo_audio}")
        resultado = self.model.transcribe(str(arquivo_audio), language="pt")
        texto_transcrito = resultado["text"]

        print(f"\n📝 Transcrição:\n", texto_transcrito)

        # Salvar a transcrição em um arquivo de texto
        caminho_saida = PASTA_TRANSCRICOES / f"{arquivo_audio.stem}_transcricao.txt"
        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(texto_transcrito)

        print(f"\n✅ Transcrição salva em: {caminho_saida}")
        return texto_transcrito
