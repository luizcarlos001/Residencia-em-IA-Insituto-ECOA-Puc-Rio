# Residência em Inteligência Artificial — Instituto ECOA PUC-Rio

Repositório com as atividades desenvolvidas durante a Residência em Inteligência Artificial do Instituto ECOA — PUC-Rio.

## Organização

```text
AULA_01/
AULA_02/
AULA_03/
AULA_04/
AULA_05/
```

## Aulas

### AULA_01 — Introdução a LLMs e APIs

Primeiros testes com modelos de linguagem utilizando Python e API.

Principais pontos:
- configuração do ambiente;
- uso de variáveis de ambiente;
- envio de prompts;
- leitura das respostas da API.

### AULA_02 — Processamento de PDFs

Uso do Docling para converter documentos PDF em Markdown e extrair informações estruturadas.

```text
PDF → Markdown → Metadados
```

### AULA_03 — Embeddings e Similaridade

Implementação e testes com embeddings.

Foram trabalhadas:
- Distância Euclidiana;
- Similaridade de Cosseno;
- Distância de Cosseno;
- busca semântica entre textos.

### AULA_04 — Chunking com LangChain

Comparação de diferentes formas de dividir documentos em chunks.

Foram testadas 10 estratégias, incluindo:
- diferentes tamanhos de chunk;
- overlap;
- divisão por parágrafos;
- divisão por sentenças;
- RecursiveCharacterTextSplitter;
- MarkdownHeaderTextSplitter.

```text
PDF → Markdown → Chunking → Embeddings → JSON
```

### AULA_05 — Documents e Busca Vetorial

Uso da estrutura `Document` do LangChain para organizar os chunks gerados anteriormente.

Também foram utilizados:
- metadados;
- Hugging Face Embeddings;
- InMemoryVectorStore;
- busca semântica;
- filtros por metadados;
- busca em múltiplos documentos.

```text
Chunks → Document → Embeddings → Vector Store → Busca
```

## Tecnologias

- Python
- LangChain
- Docling
- Hugging Face
- Sentence Transformers
- NumPy
- Pandas
- OpenAI API
- Git e GitHub

## Ambiente

Para criar e ativar um ambiente virtual no Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Para instalar as dependências:

```bash
pip install -r requirements.txt
```

## Observação

Arquivos com chaves de API, como `.env`, não devem ser enviados para o GitHub.

## Autor

Luiz Carlos Gomes da Silva Júnior
