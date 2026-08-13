from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import nltk
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"
RESULTS_DIR = BASE_DIR / "results"

# Mesmo modelo em TODOS os testes, como exige a atividade.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Os três documentos usados nas aulas anteriores podem servir como piloto.
PILOT_DOCUMENTS = {
    "bioetica_e_ia",
    "escrita_academica_ia",
    "twitter_algoritmo",
}

PORTUGUESE_DOCUMENTS = PILOT_DOCUMENTS


TEST_CONFIGS: dict[int, dict[str, Any]] = {
    1: {
        "strategy": "fixed",
        "description": "200 caracteres, sem overlap",
        "chunk_size": 200,
        "chunk_overlap": 0,
    },
    2: {
        "strategy": "fixed",
        "description": "500 caracteres, sem overlap",
        "chunk_size": 500,
        "chunk_overlap": 0,
    },
    3: {
        "strategy": "fixed",
        "description": "1000 caracteres, sem overlap",
        "chunk_size": 1000,
        "chunk_overlap": 0,
    },
    4: {
        "strategy": "fixed",
        "description": "2000 caracteres, sem overlap",
        "chunk_size": 2000,
        "chunk_overlap": 0,
    },
    5: {
        "strategy": "fixed_with_overlap",
        "description": "500 caracteres, overlap 50",
        "chunk_size": 500,
        "chunk_overlap": 50,
    },
    6: {
        "strategy": "fixed_with_overlap",
        "description": "500 caracteres, overlap 200",
        "chunk_size": 500,
        "chunk_overlap": 200,
    },
    7: {
        "strategy": "paragraph",
        "description": "Preservação de parágrafos como unidade natural",
        "chunk_size": 2000,
        "chunk_overlap": 0,
    },
    8: {
        "strategy": "sentences_3",
        "description": "3 sentenças por chunk",
        "sentences_per_chunk": 3,
        "chunk_overlap": 0,
    },
    9: {
        "strategy": "recursive",
        "description": "RecursiveCharacterTextSplitter com separadores hierárquicos",
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "separators": ["\\n\\n", "\\n", " ", ""],
    },
    10: {
        "strategy": "markdown",
        "description": "MarkdownHeaderTextSplitter por headings/seções",
        "chunk_size": None,
        "chunk_overlap": 0,
    },
}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def ensure_nltk_resources() -> None:
    """Baixa os recursos necessários ao NLTKTextSplitter."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            # Algumas versões do NLTK não usam punkt_tab.
            pass


def build_docling_converter() -> DocumentConverter:
    """
    Cria o conversor de PDF do Docling.

    A estrutura de tabelas é mantida e imagens detectadas são preservadas
    para exportação em Markdown por referência.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def convert_pdf_to_markdown(
    pdf_path: Path,
    document_dir: Path,
    converter: DocumentConverter,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    markdown_dir = document_dir / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    md_path = markdown_dir / f"{pdf_path.stem}.md"
    metadata_path = markdown_dir / "conversion_metadata.json"

    if md_path.exists() and metadata_path.exists() and not force:
        with metadata_path.open("r", encoding="utf-8") as fp:
            return md_path, json.load(fp)

    print(f"\n[DOCLING] Convertendo: {pdf_path.name}")

    result = converter.convert(pdf_path)
    doc = result.document

    # REFERENCED: imagens ficam como arquivos separados e são referenciadas no MD.
    doc.save_as_markdown(
        md_path,
        image_mode=ImageRefMode.REFERENCED,
    )

    metadata = {
        "document_id": pdf_path.stem,
        "document_name": pdf_path.name,
        "markdown_file": str(md_path.relative_to(BASE_DIR)),
        "num_pages": len(doc.pages),
        "num_tables": len(doc.tables),
        "num_pictures": len(doc.pictures),
        "conversion_tool": "Docling",
        "image_mode": "referenced",
    }
    save_json(metadata_path, metadata)

    print(
        f"  -> {metadata['num_pages']} páginas | "
        f"{metadata['num_tables']} tabelas | "
        f"{metadata['num_pictures']} imagens"
    )

    return md_path, metadata


def language_for_document(document_id: str) -> str:
    if document_id.lower() in PORTUGUESE_DOCUMENTS:
        return "portuguese"
    return "english"


def _sentence_unit_length(text: str) -> int:
    """
    Faz o NLTKTextSplitter medir cada sentença como uma unidade.
    O separador (espaço) vale zero. Assim, chunk_size=3 corresponde
    a exatamente 3 sentenças, sempre que houver 3 disponíveis.
    """
    return 0 if not text.strip() else 1


def split_text(
    test_id: int,
    markdown_text: str,
    document_id: str,
) -> list[dict[str, Any]]:
    config = TEST_CONFIGS[test_id]

    if test_id in {1, 2, 3, 4, 5, 6}:
        # separator="" força divisão estritamente por caracteres.
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            length_function=len,
            keep_separator=False,
        )
        chunks = splitter.split_text(markdown_text)
        return [{"text": chunk, "metadata": {}} for chunk in chunks if chunk.strip()]

    if test_id == 7:
        # Mantém os parágrafos inteiros sempre que possível.
        # Um parágrafo individual maior que 2000 caracteres pode permanecer maior,
        # o que é útil para observar a limitação da estratégia.
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=config["chunk_size"],
            chunk_overlap=0,
            length_function=len,
            keep_separator=False,
        )
        chunks = splitter.split_text(markdown_text)
        return [{"text": chunk, "metadata": {}} for chunk in chunks if chunk.strip()]

    if test_id == 8:
        # NLTKTextSplitter usa o tokenizer de sentenças do NLTK.
        splitter = NLTKTextSplitter(
            separator=" ",
            language=language_for_document(document_id),
            chunk_size=3,
            chunk_overlap=0,
            length_function=_sentence_unit_length,
            keep_separator=False,
        )
        chunks = splitter.split_text(markdown_text)
        return [{"text": chunk, "metadata": {}} for chunk in chunks if chunk.strip()]

    if test_id == 9:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=config["separators"],
            length_function=len,
            add_start_index=True,
        )
        docs = splitter.create_documents([markdown_text])
        return [
            {
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
            }
            for doc in docs
            if doc.page_content.strip()
        ]

    if test_id == 10:
        headers_to_split_on = [
            ("#", "heading_1"),
            ("##", "heading_2"),
            ("###", "heading_3"),
            ("####", "heading_4"),
        ]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        docs = splitter.split_text(markdown_text)
        return [
            {
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
            }
            for doc in docs
            if doc.page_content.strip()
        ]

    raise ValueError(f"Teste inválido: {test_id}")


def count_embedding_tokens(model: SentenceTransformer, text: str) -> int:
    tokenizer = model.tokenizer
    if tokenizer is None:
        return 0

    try:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        )
    except TypeError:
        return len(tokenizer.encode(text))


def create_chunk_records(
    document_id: str,
    document_name: str,
    test_id: int,
    pieces: list[dict[str, Any]],
    model: SentenceTransformer,
    markdown_file: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = TEST_CONFIGS[test_id]
    texts = [piece["text"] for piece in pieces]

    if not texts:
        return [], {
            "num_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "avg_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "embedding_dimension": 0,
            "overlapping_chunks": 0,
            "overlap_percent": 0,
        }

    print(f"    Gerando {len(texts)} embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    token_counts = [count_embedding_tokens(model, text) for text in texts]
    char_sizes = [len(text) for text in texts]

    records: list[dict[str, Any]] = []

    for index, (piece, embedding, token_count) in enumerate(
        zip(pieces, embeddings, token_counts),
        start=1,
    ):
        metadata = {
            "markdown_file": markdown_file,
            **piece.get("metadata", {}),
        }

        records.append(
            {
                "chunk_id": (
                    f"{document_id}_test{test_id:02d}_chunk{index:04d}"
                ),
                "document_id": document_id,
                "document_name": document_name,
                "test_id": test_id,
                "strategy": config["strategy"],
                "chunk_size": config.get("chunk_size"),
                "chunk_overlap": config.get("chunk_overlap", 0),
                "text": piece["text"],
                "token_count": token_count,
                "embedding": embedding.tolist(),
                "metadata": metadata,
            }
        )

    overlap = int(config.get("chunk_overlap") or 0)
    chunk_size = config.get("chunk_size")
    overlap_percent = (
        round((overlap / chunk_size) * 100, 2)
        if overlap and isinstance(chunk_size, int) and chunk_size > 0
        else 0.0
    )

    stats = {
        "num_chunks": len(records),
        "avg_chunk_size": round(statistics.mean(char_sizes), 2),
        "min_chunk_size": min(char_sizes),
        "max_chunk_size": max(char_sizes),
        "avg_tokens": round(statistics.mean(token_counts), 2),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "embedding_dimension": int(embeddings.shape[1]),
        "overlapping_chunks": max(0, len(records) - 1) if overlap > 0 else 0,
        "overlap_percent": overlap_percent,
    }

    return records, stats


def run_experiment(
    pdf_path: Path,
    md_path: Path,
    conversion_metadata: dict[str, Any],
    model: SentenceTransformer,
    test_id: int,
) -> dict[str, Any]:
    document_id = pdf_path.stem
    document_dir = RESULTS_DIR / document_id
    test_dir = document_dir / f"test_{test_id:02d}"
    output_path = test_dir / "chunks_embeddings.json"

    markdown_text = md_path.read_text(encoding="utf-8")

    print(
        f"\n  [TESTE {test_id:02d}] "
        f"{TEST_CONFIGS[test_id]['description']}"
    )

    pieces = split_text(test_id, markdown_text, document_id)

    records, stats = create_chunk_records(
        document_id=document_id,
        document_name=pdf_path.name,
        test_id=test_id,
        pieces=pieces,
        model=model,
        markdown_file=str(md_path.relative_to(BASE_DIR)),
    )

    payload = {
        "document": pdf_path.name,
        "document_id": document_id,
        "embedding_model": EMBEDDING_MODEL,
        "experiment": {
            "test_id": test_id,
            **TEST_CONFIGS[test_id],
        },
        "conversion_metadata": conversion_metadata,
        "statistics": stats,
        "chunks": records,
    }
    save_json(output_path, payload)

    print(
        f"    -> {stats['num_chunks']} chunks | "
        f"média {stats['avg_chunk_size']} caracteres | "
        f"emb. {stats['embedding_dimension']}D"
    )

    return {
        "test_id": test_id,
        **TEST_CONFIGS[test_id],
        **stats,
    }


def select_pdfs(mode: str) -> list[Path]:
    pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em: {PDF_DIR}\n"
            "Copie os PDFs da atividade para a pasta AULA_04/pdfs."
        )

    if mode == "pilot":
        selected = [
            pdf
            for pdf in pdfs
            if pdf.stem.lower() in PILOT_DOCUMENTS
        ]
        if selected:
            return selected
        return pdfs[:3]

    return pdfs


def build_aggregate_summary(
    document_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregates: dict[int, dict[str, Any]] = {}

    for test_id in TEST_CONFIGS:
        experiment_rows = []
        for doc in document_summaries:
            for exp in doc["experiments"]:
                if exp["test_id"] == test_id:
                    experiment_rows.append(exp)

        if not experiment_rows:
            continue

        total_chunks = sum(row["num_chunks"] for row in experiment_rows)
        avg_chunk_sizes = [
            row["avg_chunk_size"]
            for row in experiment_rows
            if row["num_chunks"] > 0
        ]

        aggregates[test_id] = {
            "test_id": test_id,
            "strategy": TEST_CONFIGS[test_id]["strategy"],
            "description": TEST_CONFIGS[test_id]["description"],
            "documents_processed": len(experiment_rows),
            "total_chunks": total_chunks,
            "mean_of_avg_chunk_sizes": (
                round(statistics.mean(avg_chunk_sizes), 2)
                if avg_chunk_sizes
                else 0
            ),
        }

    return {
        "embedding_model": EMBEDDING_MODEL,
        "documents": document_summaries,
        "aggregate_by_test": list(aggregates.values()),
    }


def generate_report(summary: dict[str, Any]) -> None:
    report_path = BASE_DIR / "RELATORIO_GERADO.md"

    aggregate = summary.get("aggregate_by_test", [])
    if aggregate:
        most = max(aggregate, key=lambda row: row["total_chunks"])
        least = min(aggregate, key=lambda row: row["total_chunks"])
    else:
        most = least = None

    lines = [
        "# Relatório — Avaliação de Estratégias de Chunking com LangChain",
        "",
        f"**Modelo de embeddings:** `{EMBEDDING_MODEL}`",
        "",
        "## 1. Conversão PDF → Markdown",
        "",
        "| Documento | Páginas | Tabelas | Imagens |",
        "|---|---:|---:|---:|",
    ]

    for doc in summary.get("documents", []):
        conv = doc["conversion_metadata"]
        lines.append(
            f"| {doc['document']} | "
            f"{conv.get('num_pages', 0)} | "
            f"{conv.get('num_tables', 0)} | "
            f"{conv.get('num_pictures', 0)} |"
        )

    lines += [
        "",
        "## 2. Resumo comparativo dos testes",
        "",
        "| Teste | Estratégia | Total de chunks | Média dos tamanhos médios |",
        "|---:|---|---:|---:|",
    ]

    for row in aggregate:
        lines.append(
            f"| {row['test_id']} | {row['description']} | "
            f"{row['total_chunks']} | "
            f"{row['mean_of_avg_chunk_sizes']} |"
        )

    lines += [
        "",
        "## 3. Primeiras conclusões automáticas",
        "",
    ]

    if most and least:
        lines += [
            f"- Estratégia que gerou mais chunks: **Teste {most['test_id']} — {most['description']}**, com {most['total_chunks']} chunks no total.",
            f"- Estratégia que gerou menos chunks: **Teste {least['test_id']} — {least['description']}**, com {least['total_chunks']} chunks no total.",
            "- O tamanho dos chunks pode ser comparado quantitativamente pela tabela acima e pelos `summary.json` de cada documento.",
        ]

    lines += [
        "",
        "## 4. Análise obrigatória",
        "",
        "As perguntas abaixo devem ser concluídas após inspecionar os exemplos de chunks e os Markdown gerados.",
        "",
        "1. **Qual estratégia gerou mais chunks?**",
        "   - Resposta automática disponível na seção anterior.",
        "",
        "2. **Qual gerou menos chunks?**",
        "   - Resposta automática disponível na seção anterior.",
        "",
        "3. **Como o tamanho dos chunks variou?**",
        "   - Comparar as estatísticas de tamanho médio, mínimo e máximo.",
        "",
        "4. **Qual estratégia preservou melhor a estrutura dos documentos?**",
        "   - Preencher após comparar os testes 7, 9 e 10.",
        "",
        "5. **Como tabelas foram tratadas?**",
        "   - Inspecionar o Markdown gerado pelo Docling e observar se a estrutura tabular foi mantida.",
        "",
        "6. **Como imagens foram tratadas?**",
        "   - O pipeline usa imagens referenciadas no Markdown; validar os arquivos gerados.",
        "",
        "7. **Quais informações foram perdidas durante a conversão PDF → Markdown?**",
        "   - Comparar visualmente PDF e Markdown.",
        "",
        "8. **O chunking por caracteres fragmentou conceitos ou estruturas importantes?**",
        "   - Comparar especialmente os testes 1 a 6.",
        "",
        "9. **O chunking por parágrafo produziu chunks muito grandes?**",
        "   - Conferir máximo e média do teste 7.",
        "",
        "10. **O chunking por sentença conseguiu preservar melhor o contexto?**",
        "    - Inspecionar exemplos do teste 8.",
        "",
        "11. **O Recursive Splitter apresentou vantagens?**",
        "    - Comparar o teste 9 com os testes fixos.",
        "",
        "12. **O Markdown Splitter conseguiu preservar a estrutura semântica?**",
        "    - Verificar headings armazenados nos metadados do teste 10.",
        "",
        "13. **Qual estratégia parece mais adequada para um sistema de RAG?**",
        "    - Concluir considerando contexto, tamanho, estrutura, tabelas, imagens e recuperação futura.",
        "",
        "14. **Quais estratégias devem ser descartadas?**",
        "    - Justificar com base nos resultados experimentais.",
        "",
        "15. **Quais estratégias devem ser utilizadas nos próximos experimentos?**",
        "    - Selecionar as estratégias mais equilibradas e justificar.",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aula 04 — avaliação de 10 estratégias de chunking."
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="full",
        help=(
            "pilot: usa os 3 documentos das aulas anteriores; "
            "full: processa todos os PDFs."
        ),
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Reconverte PDFs mesmo que o Markdown já exista.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ensure_nltk_resources()

    pdfs = select_pdfs(args.mode)

    print("=" * 72)
    print("AULA 04 — ESTRATÉGIAS DE CHUNKING")
    print("=" * 72)
    print(f"Modo: {args.mode}")
    print(f"PDFs selecionados: {len(pdfs)}")
    print(f"Modelo de embeddings: {EMBEDDING_MODEL}")

    converter = build_docling_converter()

    # Conversão antes de carregar o modelo de embeddings.
    converted: list[tuple[Path, Path, dict[str, Any]]] = []
    for pdf_path in pdfs:
        document_dir = RESULTS_DIR / pdf_path.stem
        try:
            md_path, conversion_metadata = convert_pdf_to_markdown(
                pdf_path,
                document_dir,
                converter,
                force=args.force_convert,
            )
            converted.append((pdf_path, md_path, conversion_metadata))
        except Exception as exc:
            print(f"[ERRO] Falha na conversão de {pdf_path.name}: {exc}")

    if not converted:
        raise RuntimeError("Nenhum documento foi convertido com sucesso.")

    print("\n[EMBEDDINGS] Carregando modelo...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    document_summaries: list[dict[str, Any]] = []

    for pdf_path, md_path, conversion_metadata in converted:
        print("\n" + "=" * 72)
        print(f"DOCUMENTO: {pdf_path.name}")
        print("=" * 72)

        experiments: list[dict[str, Any]] = []

        for test_id in range(1, 11):
            try:
                experiment_summary = run_experiment(
                    pdf_path=pdf_path,
                    md_path=md_path,
                    conversion_metadata=conversion_metadata,
                    model=model,
                    test_id=test_id,
                )
                experiments.append(experiment_summary)
            except Exception as exc:
                print(
                    f"  [ERRO] Teste {test_id:02d} falhou "
                    f"em {pdf_path.name}: {exc}"
                )
                experiments.append(
                    {
                        "test_id": test_id,
                        **TEST_CONFIGS[test_id],
                        "error": str(exc),
                    }
                )

        document_summary = {
            "document": pdf_path.name,
            "document_id": pdf_path.stem,
            "conversion_metadata": conversion_metadata,
            "embedding_model": EMBEDDING_MODEL,
            "experiments": experiments,
        }

        save_json(
            RESULTS_DIR / pdf_path.stem / "summary.json",
            document_summary,
        )
        document_summaries.append(document_summary)

    global_summary = build_aggregate_summary(document_summaries)
    save_json(RESULTS_DIR / "summary.json", global_summary)
    generate_report(global_summary)

    print("\n" + "=" * 72)
    print("CONCLUÍDO")
    print("=" * 72)
    print(f"Resultados: {RESULTS_DIR}")
    print(f"Resumo: {RESULTS_DIR / 'summary.json'}")
    print(f"Relatório: {BASE_DIR / 'RELATORIO_GERADO.md'}")


if __name__ == "__main__":
    main()
