# Relatório — Avaliação de Estratégias de Chunking com LangChain

**Modelo de embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## 1. Conversão PDF → Markdown

| Documento | Páginas | Tabelas | Imagens |
|---|---:|---:|---:|
| attention_is_all_you_need.pdf | 15 | 4 | 6 |
| bert_pretraining.pdf | 16 | 8 | 5 |
| bioetica_e_ia.pdf | 12 | 0 | 4 |
| escrita_academica_ia.pdf | 14 | 2 | 8 |
| gpt3_language_models.pdf | 75 | 51 | 34 |
| gpt4_technical_report.pdf | 100 | 12 | 29 |
| instruct_gpt.pdf | 68 | 26 | 24 |
| llama_foundation_models.pdf | 27 | 18 | 2 |
| lora_low_rank_adaptation.pdf | 26 | 18 | 8 |
| retrieval_augmented_generation.pdf | 19 | 7 | 4 |
| scaling_laws_llm.pdf | 30 | 9 | 24 |
| twitter_algoritmo.pdf | 18 | 0 | 23 |

## 2. Resumo comparativo dos testes

| Teste | Estratégia | Total de chunks | Média dos tamanhos médios |
|---:|---|---:|---:|
| 1 | 200 caracteres, sem overlap | 7444 | 194.33 |
| 2 | 500 caracteres, sem overlap | 3038 | 487.9 |
| 3 | 1000 caracteres, sem overlap | 1526 | 982.29 |
| 4 | 2000 caracteres, sem overlap | 767 | 1959.56 |
| 5 | 500 caracteres, overlap 50 | 3372 | 487.96 |
| 6 | 500 caracteres, overlap 200 | 5054 | 488.67 |
| 7 | Preservação de parágrafos como unidade natural | 765 | 1908.77 |
| 8 | 3 sentenças por chunk | 3726 | 391.16 |
| 9 | RecursiveCharacterTextSplitter com separadores hierárquicos | 1706 | 969.62 |
| 10 | MarkdownHeaderTextSplitter por headings/seções | 707 | 2387.14 |

## 3. Primeiras conclusões automáticas

- Estratégia que gerou mais chunks: **Teste 1 — 200 caracteres, sem overlap**, com 7444 chunks no total.
- Estratégia que gerou menos chunks: **Teste 10 — MarkdownHeaderTextSplitter por headings/seções**, com 707 chunks no total.
- O tamanho dos chunks pode ser comparado quantitativamente pela tabela acima e pelos `summary.json` de cada documento.

## 4. Análise obrigatória

As perguntas abaixo devem ser concluídas após inspecionar os exemplos de chunks e os Markdown gerados.

1. **Qual estratégia gerou mais chunks?**
   - Resposta automática disponível na seção anterior.

2. **Qual gerou menos chunks?**
   - Resposta automática disponível na seção anterior.

3. **Como o tamanho dos chunks variou?**
   - Comparar as estatísticas de tamanho médio, mínimo e máximo.

4. **Qual estratégia preservou melhor a estrutura dos documentos?**
   - Preencher após comparar os testes 7, 9 e 10.

5. **Como tabelas foram tratadas?**
   - Inspecionar o Markdown gerado pelo Docling e observar se a estrutura tabular foi mantida.

6. **Como imagens foram tratadas?**
   - O pipeline usa imagens referenciadas no Markdown; validar os arquivos gerados.

7. **Quais informações foram perdidas durante a conversão PDF → Markdown?**
   - Comparar visualmente PDF e Markdown.

8. **O chunking por caracteres fragmentou conceitos ou estruturas importantes?**
   - Comparar especialmente os testes 1 a 6.

9. **O chunking por parágrafo produziu chunks muito grandes?**
   - Conferir máximo e média do teste 7.

10. **O chunking por sentença conseguiu preservar melhor o contexto?**
    - Inspecionar exemplos do teste 8.

11. **O Recursive Splitter apresentou vantagens?**
    - Comparar o teste 9 com os testes fixos.

12. **O Markdown Splitter conseguiu preservar a estrutura semântica?**
    - Verificar headings armazenados nos metadados do teste 10.

13. **Qual estratégia parece mais adequada para um sistema de RAG?**
    - Concluir considerando contexto, tamanho, estrutura, tabelas, imagens e recuperação futura.

14. **Quais estratégias devem ser descartadas?**
    - Justificar com base nos resultados experimentais.

15. **Quais estratégias devem ser utilizadas nos próximos experimentos?**
    - Selecionar as estratégias mais equilibradas e justificar.
