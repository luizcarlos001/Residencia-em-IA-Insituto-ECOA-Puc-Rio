import numpy as np
import pandas as pd


def distancia_euclidiana(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula a distância euclidiana entre dois embeddings.
    """

    if len(vec1) != len(vec2):
        raise ValueError("Os embeddings devem possuir a mesma dimensão.")

    return float(np.linalg.norm(vec1 - vec2))


def similaridade_cosseno(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula a similaridade de cosseno entre dois embeddings.
    """

    if len(vec1) != len(vec2):
        raise ValueError("Os embeddings devem possuir a mesma dimensão.")

    norma_vec1 = np.linalg.norm(vec1)
    norma_vec2 = np.linalg.norm(vec2)

    if norma_vec1 == 0 or norma_vec2 == 0:
        return 0.0

    return float(
        np.dot(vec1, vec2) / (norma_vec1 * norma_vec2)
    )


def distancia_cosseno(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula a distância de cosseno.

    Distância = 1 - similaridade
    """

    return 1.0 - similaridade_cosseno(vec1, vec2)

    # ============================================================
# TESTE DAS FUNÇÕES
# ============================================================

embedding_a = np.array([1, 0, 0], dtype=np.float32)
embedding_b = np.array([0, 1, 0], dtype=np.float32)
embedding_c = np.array([1, 0, 0], dtype=np.float32)

pares = [
    ("embedding_a x embedding_b", embedding_a, embedding_b),
    ("embedding_a x embedding_c", embedding_a, embedding_c),
    ("embedding_b x embedding_c", embedding_b, embedding_c)
]

for nome, vec1, vec2 in pares:
    print(f"\n{nome}")
    print(f"Distância Euclidiana: {distancia_euclidiana(vec1, vec2):.4f}")
    print(f"Similaridade de Cosseno: {similaridade_cosseno(vec1, vec2):.4f}")
    print(f"Distância de Cosseno: {distancia_cosseno(vec1, vec2):.4f}")