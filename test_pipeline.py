import sys
from pathlib import Path
import fitz  # PyMuPDF para leitura de PDFs
import inquirer  # Biblioteca para interface interativa no terminal

# Define o diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent

# Adiciona os caminhos ao sys.path para reconhecimento dos módulos
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "app"))

# Importa os módulos corretamente
from app.rag.document_processor import DocumentProcessor
from app.rag.retriever import KnowledgeBase
from app.rag.llama_model import LLAMAModel


def extrair_texto_pdf(pdf_path):
    """ Extrai texto de um arquivo PDF usando PyMuPDF (fitz) """
    try:
        with fitz.open(pdf_path) as doc:
            texto = "\n".join([page.get_text("text") for page in doc])

        if texto.strip():
            return texto
        else:
            print(f"⚠️ Nenhum texto extraído do documento: {pdf_path} (pode ser um PDF de imagens).")
            return None
    except Exception as e:
        print(f"❌ Erro ao processar {pdf_path}: {e}")
        return None


def processar_documentos():
    """ Processa documentos armazenados na base de conhecimento usando PyMuPDF """
    doc_processor = DocumentProcessor()
    documentos_brutos = doc_processor.listar_documentos()

    if not documentos_brutos:
        print("\n📂 Nenhum documento encontrado para processamento.")
        return []

    documentos_processados = []
    for arquivo in documentos_brutos:
        print(f"\n📂 Processando documento: {arquivo.name}")

        texto_extraido = extrair_texto_pdf(arquivo)
        if texto_extraido:
            documentos_processados.append((arquivo.name, texto_extraido))
            print(f"✅ Texto extraído com sucesso do documento: {arquivo.name}")

    print(f"\n📄 {len(documentos_processados)} documentos carregados e processados com PyMuPDF.")
    return documentos_processados


def responder_pergunta(kb, pergunta):
    """ Recupera contexto e gera resposta baseada nos documentos indexados """
    print(f"\n🤖 Pergunta: {pergunta}")

    documentos_relevantes = kb.buscar_conhecimento(pergunta)

    if not documentos_relevantes:
        print("\n❌ Nenhum documento relevante encontrado para a pergunta.")
        return "⚠️ Não encontrei informações relevantes para responder."

    # Instancia o modelo LLAMA para geração de resposta
    modelo_llama = LLAMAModel()
    resposta = modelo_llama.gerar_resposta(pergunta, " ".join(documentos_relevantes))

    print("\n💡 Resposta gerada:\n", resposta)
    return resposta


def interface_perguntas(kb):
    """ Interface interativa para perguntas à IA """
    while True:
        perguntas = [
            inquirer.Text("pergunta", message="Digite sua pergunta (ou 'sair' para finalizar)")
        ]
        resposta = inquirer.prompt(perguntas)
        if resposta["pergunta"].lower() == "sair":
            print("\n👋 Encerrando sessão de perguntas.")
            break

        resposta_ia = responder_pergunta(kb, resposta["pergunta"])
        print("\n🤖 Resposta da IA:", resposta_ia)


if __name__ == "__main__":
    print("\n🚀 Iniciando Teste do Pipeline com PyMuPDF...\n")

    # Processa e estrutura os documentos
    documentos_processados = processar_documentos()

    if documentos_processados:
        print("\n📄 Documentos processados com sucesso!")
        
        # Instancia o KnowledgeBase e indexa os documentos
        kb = KnowledgeBase()
        kb.indexar_documentos()

        # Inicia a interface para perguntas
        interface_perguntas(kb)

    else:
        print("\n❌ Nenhum documento foi processado.")

    print("\n✅ Teste finalizado com sucesso!")
