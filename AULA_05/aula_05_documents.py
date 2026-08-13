from langchain_core.documents import Document
import json
from pathlib import Path

documentos = [
    Document(
        page_content="Embeddings representam textos através de vetores numéricos.",
        metadata={
            "fonte": "embeddings.md",
            "pagina": 1,
            "tipo": "teoria",
            "tema": "embeddings",
            "autor": "Luiz Carlos"
        }
    ),

    Document(
        page_content="A similaridade de cosseno permite comparar embeddings.",
        metadata={
            "fonte": "embeddings.md",
            "pagina": 2,
            "tipo": "teoria",
            "tema": "embeddings",
            "autor": "Luiz Carlos"
        }
    ),

    Document(
        page_content="Chunking é o processo de dividir documentos em partes menores.",
        metadata={
            "fonte": "chunking.md",
            "pagina": 1,
            "tipo": "teoria",
            "tema": "chunking",
            "autor": "Luiz Carlos"
        }
    ),

    Document(
        page_content="O overlap permite compartilhar informações entre chunks consecutivos.",
        metadata={
            "fonte": "chunking.md",
            "pagina": 2,
            "tipo": "teoria",
            "tema": "chunking",
            "autor": "Luiz Carlos"
        }
    ),

    Document(
        page_content="RAG combina recuperação de informações com modelos de linguagem.",
        metadata={
            "fonte": "rag.md",
            "pagina": 1,
            "tipo": "teoria",
            "tema": "RAG",
            "autor": "Luiz Carlos"
        }
    )
]


for i, documento in enumerate(documentos, start=1):
    print(f"\nDocumento {i}")
    print("Conteúdo:", documento.page_content)
    print("Metadados:", documento.metadata)


print("\nQuantidade de documentos:", len(documentos))

print("\n--- EXERCÍCIO 2 ---")

schema_metadata = {
    "fonte": "attention_is_all_you_need.md",
    "documento_id": "attention_is_all_you_need",
    "chunk_index": 0,
    "estrategia": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "n_caracteres": 982,

    # Campos extras
    "pagina": 1,
    "secao": "Introduction",
    "n_tokens": 215
}

print("\nExemplo de metadados:")
print(schema_metadata)

documento_exemplo = Document(
    page_content="O Transformer utiliza mecanismos de atenção para processar sequências.",
    metadata=schema_metadata
)

print("\nDocumento com schema completo:")
print("Conteúdo:", documento_exemplo.page_content)
print("Metadados:", documento_exemplo.metadata)

print("\n--- CHUNK REAL DA AULA 04 ---")

caminho_json = Path(
    "../AULA_04/results/attention_is_all_you_need/test_09/chunks_embeddings.json"
)

with open(caminho_json, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)

# Primeiro chunk real do teste 09
chunk = dados["chunks"][0]

metadata_real = {
    "fonte": Path(chunk["metadata"]["markdown_file"]).name,
    "documento_id": chunk["document_id"],
    "chunk_index": 0,
    "estrategia": chunk["strategy"],
    "chunk_size": chunk["chunk_size"],
    "chunk_overlap": chunk["chunk_overlap"],
    "n_caracteres": len(chunk["text"]),

    # Campos próprios
    "pagina": chunk["metadata"].get("page"),
    "secao": chunk["metadata"].get("section"),
    "n_tokens": chunk["token_count"]
}

documento_real = Document(
    page_content=chunk["text"],
    metadata=metadata_real
)

print("\nConteúdo do chunk:")
print(documento_real.page_content)

print("\nMetadados:")
print(documento_real.metadata)

print("\nExemplo em JSON:")
print(
    json.dumps(
        {
            "page_content": documento_real.page_content,
            "metadata": documento_real.metadata
        },
        ensure_ascii=False,
        indent=4
    )
)