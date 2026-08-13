# AULA_05 — Documents, Metadados e Busca Vetorial com LangChain

Nesta aula foi realizada a migração dos chunks produzidos na Aula 04 para o formato padrão `Document` do LangChain.

Também foram explorados:

- criação manual de objetos `Document`;
- definição de schema de metadados;
- reaproveitamento dos chunks reais da Aula 04;
- conversão automática de chunks para `Document`;
- geração de embeddings com Hugging Face;
- criação de uma vector store;
- busca semântica;
- busca semântica em múltiplos documentos;
- aplicação de filtros utilizando metadados.

---

## Estrutura da pasta

```text
AULA_05/
├── aula_05_documents.py
├── aula_05_chunks.py
├── aula_05_busca_vetorial.py
├── aula_05_multiplos_documentos.py
└── README.md
```

---

# Exercício 1 — Criando Documents manualmente

O primeiro exercício teve como objetivo compreender a estrutura padrão `Document` utilizada pelo LangChain.

Foram criados cinco objetos `Document` manualmente utilizando temas estudados durante o curso, como:

- embeddings;
- chunking;
- overlap;
- RAG.

Exemplo:

```python
from langchain_core.documents import Document

documento = Document(
    page_content="Embeddings representam textos através de vetores numéricos.",
    metadata={
        "fonte": "embeddings.md",
        "pagina": 1,
        "tipo": "teoria",
        "tema": "embeddings",
        "autor": "Luiz Carlos"
    }
)
```

Um `Document` possui como principais informações:

```text
page_content
metadata
```

O campo `page_content` armazena o conteúdo textual.

O campo `metadata` armazena informações adicionais relacionadas ao documento.

Ao todo foram criados:

```text
5 Documents
```

Também foi realizado um `print` exibindo o conteúdo e os metadados de cada documento.

---

## Tipos de dados aceitos em metadata

O campo `metadata` funciona como um dicionário Python e pode armazenar diferentes tipos de dados.

Entre eles:

- strings;
- números;
- booleanos;
- listas;
- dicionários.

Exemplo utilizando uma lista e um dicionário aninhado:

```python
metadata={
    "tema": "RAG",
    "palavras_chave": ["RAG", "retrieval", "LLM"],
    "informacoes": {
        "nivel": "introdutorio",
        "aula": 5
    }
}
```

O `Document` consegue armazenar essas estruturas normalmente.

Entretanto, algumas vector stores podem possuir restrições próprias sobre os tipos de metadados aceitos durante a indexação.

---

## Document sem metadata

Também foi testada a criação de um `Document` sem informar explicitamente o campo `metadata`.

Exemplo:

```python
documento = Document(
    page_content="Documento criado sem metadados."
)
```

Nesse caso, o LangChain cria automaticamente um dicionário vazio:

```python
{}
```

Portanto, não é obrigatório informar metadados no momento da criação do `Document`.

---

# Exercício 2 — Projeto do Schema de Metadados

O segundo exercício teve como objetivo definir um schema de metadados para representar os chunks gerados na Aula 04.

O schema definido foi:

| Campo | Descrição |
|---|---|
| `fonte` | Nome do arquivo Markdown de origem |
| `documento_id` | Identificador do documento |
| `chunk_index` | Posição do chunk dentro do documento |
| `estrategia` | Estratégia de chunking utilizada |
| `chunk_size` | Tamanho configurado para o chunk |
| `chunk_overlap` | Sobreposição configurada |
| `n_caracteres` | Número real de caracteres do chunk |
| `pagina` | Página de origem do conteúdo |
| `secao` | Seção do documento |
| `n_tokens` | Quantidade de tokens do chunk |

---

## Campos adicionais escolhidos

Além dos campos obrigatórios, foram adicionados três novos campos:

### `pagina`

Permite identificar a página original de onde determinada informação foi extraída.

Esse campo é importante principalmente para sistemas RAG, pois permite citar a origem da informação recuperada.

Exemplo de pergunta que esse campo permite responder:

```text
Em qual página do documento original essa informação aparece?
```

---

### `secao`

Permite identificar a seção do documento de onde o trecho foi extraído.

Exemplos:

```text
Introduction
Methods
Results
Conclusion
```

Isso pode ser utilizado posteriormente para criar filtros de busca.

Exemplo:

```text
Recupere apenas informações da seção de resultados.
```

---

### `n_tokens`

Armazena a quantidade de tokens existente em cada chunk.

Esse campo permite analisar o tamanho real dos chunks em relação ao processamento de modelos de linguagem.

Também pode ajudar a responder perguntas como:

```text
Quais estratégias de chunking produziram trechos maiores em quantidade de tokens?
```

---

# Exemplo real de chunk da Aula 04

Para testar o schema, foi utilizado um chunk real produzido na Aula 04.

O documento utilizado foi:

```text
attention_is_all_you_need.md
```

e a estratégia utilizada foi:

```text
RecursiveCharacterTextSplitter
```

O chunk foi convertido para o formato `Document` do LangChain.

Exemplo dos metadados obtidos:

```json
{
    "fonte": "attention_is_all_you_need.md",
    "documento_id": "attention_is_all_you_need",
    "chunk_index": 0,
    "estrategia": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "n_caracteres": 993,
    "pagina": null,
    "secao": null,
    "n_tokens": 257
}
```

Os valores:

```text
pagina: null
secao: null
```

não representam erro.

Eles aparecem dessa forma porque essas informações não estavam disponíveis nos metadados originais daquele chunk produzido na Aula 04.

---

# Qual campo utilizar para citar a fonte?

Para permitir que um sistema RAG informe exatamente de onde uma informação foi recuperada, os principais campos seriam:

```text
fonte
pagina
secao
```

Por exemplo, futuramente uma resposta poderia apresentar:

```text
Fonte: attention_is_all_you_need.md
Página: 2
Seção: Introduction
```

Dessa forma, o usuário consegue rastrear a origem da informação utilizada na resposta.

---

# Por que o chunk_index é útil?

O campo `chunk_index` identifica a posição de determinado chunk dentro do documento.

Por exemplo:

```text
chunk 24
chunk 25
chunk 26
```

Imagine que o sistema encontre o `chunk 25`, mas a explicação tenha sido cortada no meio.

Conhecendo o `chunk_index`, é possível recuperar também:

```text
chunk 24
chunk 26
```

Isso permite obter o contexto anterior e posterior ao trecho encontrado.

Portanto, o `chunk_index` é útil para reconstruir o contexto ao redor de uma informação recuperada.

---

# Conversão dos chunks da Aula 04 para Document

Na Aula 04, cada chunk era representado manualmente por uma estrutura semelhante a:

```json
{
    "text": "Conteúdo do chunk...",
    "embedding": [0.0123, -0.0345],
    "metadata": {}
}
```

Na Aula 05, essa estrutura foi migrada para o formato padrão do LangChain:

```python
Document(
    page_content="Conteúdo do chunk...",
    metadata={}
)
```

Uma diferença importante é que o embedding não faz parte do `Document`.

Ou seja:

```text
Document
├── page_content
└── metadata
```

O vetor de embedding fica sob responsabilidade da vector store.

---

# Conversão automática dos chunks

O arquivo:

```text
aula_05_chunks.py
```

foi utilizado para carregar automaticamente os chunks reais da Aula 04.

O fluxo utilizado foi:

```text
JSON da Aula 04
        ↓
Leitura dos chunks
        ↓
Criação do metadata
        ↓
Document
        ↓
Lista de Documents
```

Exemplo simplificado:

```python
documento = Document(
    page_content=chunk["text"],
    metadata={
        "fonte": "attention_is_all_you_need.md",
        "documento_id": "attention_is_all_you_need",
        "chunk_index": indice,
        "estrategia": "recursive"
    }
)
```

Dessa forma, não é necessário criar os documentos manualmente um por um.

---

# Modelo de Embeddings

Para gerar os embeddings foi utilizado o modelo:

```text
sentence-transformers/all-MiniLM-L6-v2
```

através da integração do LangChain com Hugging Face.

Importação:

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

Configuração:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)
```

Nesse caso, o modelo transforma o conteúdo dos objetos `Document` em representações vetoriais.

---

# Vector Store

Para armazenar os vetores foi utilizada a:

```text
InMemoryVectorStore
```

Importação:

```python
from langchain_core.vectorstores import InMemoryVectorStore
```

Criação:

```python
vector_store = InMemoryVectorStore(
    embedding=embeddings
)
```

Depois, os documentos foram adicionados:

```python
vector_store.add_documents(documentos)
```

Nesse momento a vector store gera os embeddings dos documentos e os mantém armazenados internamente.

O `Document` continua contendo apenas:

```text
page_content
metadata
```

---

# Busca Semântica

Após criar a vector store, foi realizada uma busca semântica.

A consulta utilizada foi:

```python
query = "What is the Transformer architecture?"
```

A busca foi feita utilizando:

```python
resultados = vector_store.similarity_search(
    query=query,
    k=3
)
```

O parâmetro:

```text
k=3
```

indica que devem ser retornados os três chunks mais semelhantes à consulta.

Os resultados recuperados apresentaram conteúdos relacionados a:

- Transformer;
- attention;
- multi-head attention;
- encoder;
- decoder.

Isso demonstrou que a busca vetorial conseguiu recuperar conteúdos semanticamente relacionados à pergunta, mesmo sem realizar uma busca exata por palavras.

---

# Busca com filtro de metadados

Também foi realizada uma busca utilizando filtros sobre os metadados.

Exemplo:

```python
resultados_filtrados = vector_store.similarity_search(
    query="What is the Transformer architecture?",
    k=3,
    filter=lambda doc: (
        doc.metadata.get("estrategia") == "recursive"
    )
)
```

Nesse caso, além da similaridade semântica, a busca considera apenas documentos cujo campo:

```text
estrategia
```

possui o valor:

```text
recursive
```

---

# Busca em múltiplos documentos

Posteriormente foram carregados os chunks de vários documentos produzidos na Aula 04.

Foi utilizada a estratégia:

```text
Teste 09 — RecursiveCharacterTextSplitter
```

Os chunks de todos os documentos foram convertidos para objetos `Document` e adicionados em uma única vector store.

O fluxo ficou:

```text
Attention Is All You Need ─┐
BERT                       │
GPT-3                      │
GPT-4                      │
InstructGPT                │
LLaMA                      │
LoRA                       ├──→ Vector Store
RAG                        │
Scaling Laws               │
Bioética                   │
Escrita Acadêmica          │
Twitter/X                  │
                           ↓
                    Busca Semântica
```

---

# Busca sobre RAG

Uma das consultas utilizadas foi:

```python
query = "What is retrieval augmented generation?"
```

A vector store conseguiu recuperar corretamente trechos relacionados ao artigo:

```text
retrieval_augmented_generation.md
```

Um dos resultados apresentava conteúdo diretamente relacionado ao funcionamento de RAG e mecanismos de retrieval.

Isso demonstra que a vector store conseguiu identificar semanticamente o documento mais relacionado à consulta.

---

# Busca com filtro por documento

Também foi realizado um filtro utilizando o campo:

```text
documento_id
```

Exemplo:

```python
resultados_filtrados = vector_store.similarity_search(
    query="What is retrieval augmented generation?",
    k=3,
    filter=lambda doc: (
        doc.metadata.get("documento_id")
        == "retrieval_augmented_generation"
    )
)
```

Nesse caso, a busca é limitada aos chunks pertencentes ao documento:

```text
retrieval_augmented_generation
```

Os resultados retornados apresentaram:

```text
Fonte: retrieval_augmented_generation.md
```

Isso demonstra que é possível combinar:

```text
Busca Semântica
        +
Filtro de Metadados
```

para controlar quais documentos podem participar da recuperação.

---

# Exemplo de resultado filtrado

Um dos chunks recuperados apresentou metadados semelhantes a:

```json
{
    "fonte": "retrieval_augmented_generation.md",
    "documento_id": "retrieval_augmented_generation",
    "chunk_index": 4,
    "estrategia": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "n_caracteres": 995,
    "pagina": null,
    "secao": null,
    "n_tokens": 311
}
```

O conteúdo recuperado tratava diretamente de Retrieval-Augmented Generation e da utilização de memória paramétrica e não paramétrica.

---

# Fluxo final da Aula 05

O pipeline construído nesta aula ficou:

```text
Chunks da Aula 04
        ↓
Document
        ↓
Metadata
        ↓
HuggingFaceEmbeddings
        ↓
InMemoryVectorStore
        ↓
Busca Semântica
        ↓
Filtro por Metadata
        ↓
Chunks mais relevantes
```

---

# Arquivos desenvolvidos

## `aula_05_documents.py`

Responsável por:

- criar `Document` manualmente;
- imprimir `page_content`;
- imprimir `metadata`;
- testar diferentes tipos de metadata;
- definir o schema;
- utilizar um chunk real da Aula 04.

---

## `aula_05_chunks.py`

Responsável por:

- carregar os JSONs da Aula 04;
- percorrer os chunks;
- construir os metadados;
- converter os chunks automaticamente para `Document`.

---

## `aula_05_busca_vetorial.py`

Responsável por:

- carregar os chunks;
- criar os `Document`;
- configurar `HuggingFaceEmbeddings`;
- criar a `InMemoryVectorStore`;
- adicionar os documentos;
- realizar busca semântica;
- realizar busca com filtro.

---

## `aula_05_multiplos_documentos.py`

Responsável por:

- localizar os resultados de vários documentos da Aula 04;
- carregar os chunks do Teste 09;
- converter todos para `Document`;
- armazenar os documentos em uma única vector store;
- realizar busca semântica entre vários artigos;
- filtrar resultados utilizando `documento_id`.

---

# Dependências

As principais bibliotecas utilizadas foram:

```text
langchain-core
langchain-text-splitters
sentence-transformers
langchain-huggingface
```

Instalação:

```powershell
python -m pip install langchain-core langchain-text-splitters sentence-transformers langchain-huggingface
```

---

# Como executar

Executar o Exercício 1 e o schema:

```powershell
python .\aula_05_documents.py
```

Executar a conversão automática dos chunks:

```powershell
python .\aula_05_chunks.py
```

Executar a primeira busca vetorial:

```powershell
python .\aula_05_busca_vetorial.py
```

Executar a busca utilizando múltiplos documentos:

```powershell
python .\aula_05_multiplos_documentos.py
```

---

# Conclusão

Nesta aula foi possível migrar as estruturas de chunks criadas manualmente na Aula 04 para o formato padrão `Document` utilizado pelo ecossistema LangChain.

A principal separação observada foi:

```text
Document
↓
Texto + Metadados
```

enquanto:

```text
Vector Store
↓
Embeddings + Armazenamento + Busca
```

Também foi possível utilizar os chunks reais gerados anteriormente e indexá-los em uma vector store.

A busca vetorial permitiu recuperar trechos semanticamente relacionados às consultas realizadas, enquanto os metadados permitiram aplicar filtros sobre os resultados.

Dessa forma, a atividade estabelece a base necessária para construção de sistemas mais completos de recuperação de informação e Retrieval-Augmented Generation (RAG).