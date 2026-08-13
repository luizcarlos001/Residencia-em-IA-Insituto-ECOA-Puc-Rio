import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================================
# 1. LER OS CHUNKS REAIS DA AULA 04
# ==========================================================

caminho_json = Path(
    "../AULA_04/results/attention_is_all_you_need/test_09/chunks_embeddings.json"
)

with open(caminho_json, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)


documentos = []

for indice, chunk in enumerate(dados["chunks"]):

    metadata = {
        "fonte": Path(chunk["metadata"]["markdown_file"]).name,
        "documento_id": chunk["document_id"],
        "chunk_index": indice,
        "estrategia": chunk["strategy"],
        "chunk_size": chunk["chunk_size"],
        "chunk_overlap": chunk["chunk_overlap"],
        "n_caracteres": len(chunk["text"]),
        "pagina": chunk["metadata"].get("page"),
        "secao": chunk["metadata"].get("section"),
        "n_tokens": chunk["token_count"]
    }

    documentos.append(
        Document(
            page_content=chunk["text"],
            metadata=metadata
        )
    )


print("Documents carregados:", len(documentos))


# ==========================================================
# 2. MODELO DE EMBEDDING
# ==========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Modelo de embeddings carregado.")


# ==========================================================
# 3. CRIAR VECTOR STORE
# ==========================================================

vector_store = InMemoryVectorStore(
    embedding=embeddings
)

vector_store.add_documents(documentos)

print("Documents adicionados à vector store.")


# ==========================================================
# 4. BUSCA SEMÂNTICA
# ==========================================================

query = "What is the Transformer architecture?"

resultados = vector_store.similarity_search(
    query=query,
    k=3
)


print("\n--- RESULTADOS DA BUSCA ---")

for i, documento in enumerate(resultados, start=1):

    print(f"\nResultado {i}")

    print("\nConteúdo:")
    print(documento.page_content)

    print("\nMetadados:")
    print(documento.metadata)

    print("\n--- BUSCA COM FILTRO DE METADADOS ---")

resultados_filtrados = vector_store.similarity_search(
    query="What is the Transformer architecture?",
    k=3,
    filter=lambda doc: doc.metadata.get("estrategia") == "recursive"
)

for i, documento in enumerate(resultados_filtrados, start=1):
    print(f"\nResultado filtrado {i}")

    print("\nConteúdo:")
    print(documento.page_content)

    print("\nMetadados:")
    print(documento.metadata)