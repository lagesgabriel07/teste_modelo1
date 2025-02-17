import logging
from pathlib import Path
from typing import Optional

# Configuração do Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class AudioHandler:
    def __init__(self, pasta_audio: str = "data/audios"):
        """
        Inicializa o manipulador de arquivos de áudio.
        
        Args:
            pasta_audio (str): Caminho do diretório onde os áudios serão armazenados.
        """
        self.pasta_audio = Path(pasta_audio)
        self._verificar_criar_diretorio()

    def _verificar_criar_diretorio(self):
        """Verifica se o diretório de áudio existe, e se não, cria automaticamente."""
        if not self.pasta_audio.exists():
            self.pasta_audio.mkdir(parents=True, exist_ok=True)
            logger.info(f"Diretório '{self.pasta_audio}' criado.")

    def listar_arquivos_audio(self) -> Optional[Path]:
        """
        Lista os arquivos de áudio disponíveis na pasta e retorna o mais recente.

        Returns:
            Path | None: Caminho do arquivo mais recente ou None se a pasta estiver vazia.
        """
        extensoes_validas = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}

        arquivos = sorted(
            [f for f in self.pasta_audio.iterdir() if f.suffix.lower() in extensoes_validas],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        if arquivos:
            logger.info(f"Último arquivo encontrado: {arquivos[0]}")
            return arquivos[0]
        else:
            logger.warning("Nenhum arquivo de áudio encontrado no diretório.")
            return None

    def contar_arquivos_audio(self) -> int:
        """
        Conta quantos arquivos de áudio existem no diretório.

        Returns:
            int: Número de arquivos de áudio encontrados.
        """
        extensoes_validas = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        arquivos = [f for f in self.pasta_audio.iterdir() if f.suffix.lower() in extensoes_validas]

        logger.info(f"Total de arquivos de áudio encontrados: {len(arquivos)}")
        return len(arquivos)

    def salvar_audio(self, nome_arquivo: str, conteudo: bytes) -> Path:
        """
        Salva um arquivo de áudio na pasta definida.

        Args:
            nome_arquivo (str): Nome do arquivo a ser salvo.
            conteudo (bytes): Conteúdo do arquivo em bytes.

        Returns:
            Path: Caminho completo do arquivo salvo.
        """
        caminho_arquivo = self.pasta_audio / nome_arquivo
        try:
            with caminho_arquivo.open("wb") as f:
                f.write(conteudo)
            logger.info(f"Arquivo salvo com sucesso: {caminho_arquivo}")
            return caminho_arquivo
        except Exception as e:
            logger.error(f"Erro ao salvar o arquivo {nome_arquivo}: {str(e)}")
            raise e
