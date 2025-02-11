from fastapi import FastAPI
from app.audio_processing.transcriber import AudioTranscriber
from rag.retriever import KnowledgeBase
from rag.llama_model import LLAMAModel

app = FastAPI()

# Inicializa os módulos
transcriber = AudioTranscriber(modelo="medium")
kb = KnowledgeBase()
kb.indexar_documentos()
llama = LLAMAModel()

@app.post("/analisar_audio/")
async def analisar_audio(file_path: str):
    """ Recebe um arquivo de áudio, transcreve e gera uma análise """
    texto_transcrito = transcriber.transcrever_audio(file_path)
    documentos_relevantes = kb.buscar_conhecimento(texto_transcrito)
    contexto = " ".join(documentos_relevantes)
    resposta = llama.gerar_resposta(texto_transcrito, contexto)

    return {"transcricao": texto_transcrito, "resposta": resposta}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
