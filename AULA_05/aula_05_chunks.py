import json
from pathlib import Path

from langchain_core.documents import Document


# Caminho para um dos resultados reais da Aula 04
caminho_json = Path(
    "../AULA_04/results/attention_is_all_you_need/test_09/chunks_embeddings.json"
)


# Ler o JSON
with open(caminho_json, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)


# Lista onde serão armazenados os Documents
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

    documento = Document(
        page_content=chunk["text"],
        metadata=metadata
    )

    documentos.append(documento)


print("Quantidade de Documents criados:", len(documentos))


# Mostrar os 3 primeiros
for i, documento in enumerate(documentos[:3], start=1):

    print(f"\n--- DOCUMENTO {i} ---")

    print("\nConteúdo:")
    print(documento.page_content)

    print("\nMetadados:")
    print(documento.metadata)