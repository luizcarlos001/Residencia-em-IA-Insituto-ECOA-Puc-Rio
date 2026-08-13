import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================================
# 1. LOCALIZAR TODOS OS TESTES 09 DA AULA 04
# ==========================================================

pasta_results = Path("../AULA_04/results")

arquivos_json = list(
    pasta_results.glob("*/test_09/chunks_embeddings.json")
)

print("Arquivos encontrados:", len(arquivos_json))


# ==========================================================
# 2. TRANSFORMAR TODOS OS CHUNKS EM DOCUMENT
# ==========================================================

documentos = []

for caminho_json in arquivos_json:

    with open(caminho_json, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    for indice, chunk in enumerate(dados["chunks"]):

        metadata = {
            "fonte": Path(
                chunk["metadata"]["markdown_file"]
            ).name,

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


print("Total de Documents:", len(documentos))


# ==========================================================
# 3. MODELO DE EMBEDDINGS
# ==========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# ==========================================================
# 4. VECTOR STORE
# ==========================================================

vector_store = InMemoryVectorStore(
    embedding=embeddings
)

vector_store.add_documents(documentos)

print("Vector store criada.")


# ==========================================================
# 5. BUSCA EM TODOS OS DOCUMENTOS
# ==========================================================

query = "What is retrieval augmented generation?"

resultados = vector_store.similarity_search(
    query=query,
    k=5
)


print("\n--- BUSCA EM TODOS OS DOCUMENTOS ---")

for i, documento in enumerate(resultados, start=1):

    print(f"\nResultado {i}")

    print("Fonte:", documento.metadata["fonte"])
    print(
        "Chunk:",
        documento.metadata["chunk_index"]
    )

    print("\nConteúdo:")
    print(documento.page_content[:700])

    print("\n" + "-" * 70)

    print("\n--- BUSCA COM FILTRO POR DOCUMENTO ---")

resultados_filtrados = vector_store.similarity_search(
    query="What is retrieval augmented generation?",
    k=3,
    filter=lambda doc: (
        doc.metadata.get("documento_id")
        == "retrieval_augmented_generation"
    )
)

for i, documento in enumerate(resultados_filtrados, start=1):

    print(f"\nResultado filtrado {i}")

    print("Fonte:", documento.metadata["fonte"])
    print("Chunk:", documento.metadata["chunk_index"])

    print("\nConteúdo:")
    print(documento.page_content[:700])

    print("\nMetadados:")
    print(documento.metadata)

    print("\n" + "-" * 70)