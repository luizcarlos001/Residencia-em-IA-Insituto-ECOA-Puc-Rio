# AULA_04 — Estratégias de Chunking com LangChain

Esta pasta implementa o pipeline solicitado na atividade:

```text
PDF
 ↓
Docling
 ↓
Markdown
 ↓
10 estratégias de chunking
 ↓
Embeddings
 ↓
JSON
```

## Estrutura

```text
AULA_04/
├── aula_04_chunking.py
├── requirements.txt
├── pdfs/
│   └── coloque_os_pdfs_aqui.txt
├── results/
└── RELATORIO_GERADO.md  # criado após executar
```

## Estratégias implementadas

1. 200 caracteres, overlap 0
2. 500 caracteres, overlap 0
3. 1000 caracteres, overlap 0
4. 2000 caracteres, overlap 0
5. 500 caracteres, overlap 50
6. 500 caracteres, overlap 200
7. Por parágrafo
8. 3 sentenças por chunk
9. RecursiveCharacterTextSplitter
10. MarkdownHeaderTextSplitter

## 1. Instalação

No PowerShell, dentro de `AULA_04`:

```powershell
python -m pip install -r requirements.txt
```

## 2. Coloque os PDFs em `pdfs`

Exemplo:

```text
AULA_04/
└── pdfs/
    ├── attention_is_all_you_need.pdf
    ├── bert_pretraining.pdf
    ├── bioetica_e_ia.pdf
    └── ...
```

## 3. Teste rápido com 3 documentos

```powershell
python .\aula_04_chunking.py --mode pilot
```

O modo piloto procura:

- `bioetica_e_ia.pdf`
- `escrita_academica_ia.pdf`
- `twitter_algoritmo.pdf`

## 4. Execução completa

```powershell
python .\aula_04_chunking.py --mode full
```

## 5. Reconverter PDFs

Por padrão, o Markdown existente é reutilizado. Para forçar uma nova conversão:

```powershell
python .\aula_04_chunking.py --mode full --force-convert
```

## Saídas

Para cada documento:

```text
results/
└── documento/
    ├── markdown/
    │   ├── documento.md
    │   └── conversion_metadata.json
    ├── test_01/
    │   └── chunks_embeddings.json
    ├── ...
    ├── test_10/
    │   └── chunks_embeddings.json
    └── summary.json
```

Também são criados:

```text
results/summary.json
RELATORIO_GERADO.md
```

O relatório já traz as estatísticas quantitativas. As perguntas qualitativas devem ser concluídas após inspecionar os chunks e comparar os PDFs com os Markdown gerados.
