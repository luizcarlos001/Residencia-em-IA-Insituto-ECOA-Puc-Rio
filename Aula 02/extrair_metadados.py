import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


PASTA_AULA = Path("aula_2")
URL_OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
LIMITE_CARACTERES = 30000


def extrair_metadados(conteudo, api_key, modelo):
    schema = {
        "type": "object",
        "properties": {
            "titulo": {
                "type": "string"
            },
            "autores": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "ano": {
                "type": ["integer", "null"]
            }
        },
        "required": ["titulo", "autores", "ano"],
        "additionalProperties": False
    }

    prompt = f"""
Analise o documento acadêmico abaixo e extraia:

- título completo do trabalho;
- lista de autores;
- ano de publicação.

Regras:
- Não invente informações.
- Não traduza o título.
- Não use autores encontrados nas referências bibliográficas.
- Caso o ano não seja identificado com segurança, retorne null.

DOCUMENTO:

{conteudo[:LIMITE_CARACTERES]}
"""

    resposta = requests.post(
        URL_OPENROUTER,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": modelo,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Você extrai metadados de documentos acadêmicos "
                        "e responde somente no formato JSON solicitado."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "metadados_documento",
                    "strict": True,
                    "schema": schema
                }
            },
            "provider": {
                "require_parameters": True
            }
        },
        timeout=180
    )

    if not resposta.ok:
        raise RuntimeError(
            f"Erro {resposta.status_code}: {resposta.text}"
        )

    conteudo_resposta = (
        resposta.json()["choices"][0]["message"]["content"]
    )

    return json.loads(conteudo_resposta)


def main():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    modelo = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-4o-mini"
    )

    if not api_key:
        print("A chave OPENROUTER_API_KEY não foi encontrada no arquivo .env.")
        return

    arquivos_md = sorted(PASTA_AULA.glob("*.md"))

    if not arquivos_md:
        print("Nenhum arquivo Markdown foi encontrado.")
        return

    print(f"Arquivos Markdown encontrados: {len(arquivos_md)}")
    print(f"Modelo utilizado: {modelo}")

    for arquivo_md in arquivos_md:
        try:
            print(f"\nProcessando: {arquivo_md.name}")

            conteudo = arquivo_md.read_text(encoding="utf-8")

            metadados = extrair_metadados(
                conteudo,
                api_key,
                modelo
            )

            arquivo_json = (
                arquivo_md.parent
                / f"output_{arquivo_md.stem}.json"
            )

            arquivo_json.write_text(
                json.dumps(
                    metadados,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            print(
                json.dumps(
                    metadados,
                    ensure_ascii=False,
                    indent=2
                )
            )

            print(f"Salvo: {arquivo_json.name}")

        except Exception as erro:
            print(f"Erro em {arquivo_md.name}: {erro}")

    print("\nExtração finalizada.")


if __name__ == "__main__":
    main()