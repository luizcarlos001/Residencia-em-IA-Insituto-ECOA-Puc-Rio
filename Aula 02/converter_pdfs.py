import os
import traceback
from pathlib import Path

# Precisa ser definido antes de importar PyTorch/Docling
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True

from docling.document_converter import DocumentConverter


PASTA_AULA = Path("aula_2")


def main():
    if not PASTA_AULA.exists():
        print("A pasta aula_2 não foi encontrada.")
        return

    arquivos_pdf = sorted(PASTA_AULA.glob("*.pdf"))

    if not arquivos_pdf:
        print("Nenhum PDF foi encontrado na pasta aula_2.")
        return

    converter = DocumentConverter()

    convertidos = 0

    for arquivo_pdf in arquivos_pdf:
        try:
            print(f"\nConvertendo: {arquivo_pdf.name}")

            resultado = converter.convert(arquivo_pdf)

            markdown = resultado.document.export_to_markdown()

            arquivo_saida = arquivo_pdf.with_suffix(".md")

            arquivo_saida.write_text(
                markdown,
                encoding="utf-8"
            )

            print(f"Salvo: {arquivo_saida.name}")
            convertidos += 1

        except Exception as erro:
            print(f"\nErro em {arquivo_pdf.name}: {erro}")
            traceback.print_exc()

    print("\nProcessamento finalizado.")
    print(f"PDFs convertidos: {convertidos}/{len(arquivos_pdf)}")


if __name__ == "__main__":
    main()