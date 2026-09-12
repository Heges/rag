"""Консольный поиск ближайших чанков в сохранённом FAISS-индексе."""

from __future__ import annotations

import argparse
import json
import time

import faiss
import numpy as np
import ollama

from build_index import (
    CHUNKS_PATH,
    EXPECTED_DIMENSION,
    INDEX_PATH,
    INFO_PATH,
    OLLAMA_BASE_URL,
    read_faiss_index,
)


def load_artifacts() -> tuple[faiss.Index, list[dict[str, object]], dict[str, object]]:
    """Загрузить индекс, чанки и параметры, проверив их согласованность."""

    index = read_faiss_index(INDEX_PATH)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))

    if index.ntotal != len(chunks) or index.ntotal != info["chunk_count"]:
        raise ValueError("Количество векторов не совпадает с количеством метаданных")
    if index.d != EXPECTED_DIMENSION or index.d != info["dimensions"]:
        raise ValueError("Размерность индекса не совпадает с параметрами модели")
    return index, chunks, info


def search_with_trace(
    query: str,
    top_k: int = 3,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Выполнить поиск и вернуть результаты вместе с наблюдаемыми измерениями."""

    if not query.strip():
        raise ValueError("Поисковый запрос не должен быть пустым")
    if top_k < 1:
        raise ValueError("top_k должен быть положительным числом")

    index, chunks, info = load_artifacts()
    # Для локального сервиса отключаем наследование HTTP_PROXY/HTTPS_PROXY.
    client = ollama.Client(host=OLLAMA_BASE_URL, trust_env=False)
    embedding_input = f"search_query: {query.strip()}"
    embedding_started = time.perf_counter()
    response = client.embed(
        model=str(info["model"]),
        input=embedding_input,
        truncate=False,
        keep_alive="10m",
    )
    embedding_seconds = time.perf_counter() - embedding_started
    query_vector = np.asarray(response.embeddings, dtype=np.float32)
    if query_vector.shape != (1, EXPECTED_DIMENSION):
        raise ValueError(f"Неожиданная форма вектора запроса: {query_vector.shape}")

    # Индекс содержит нормализованные документы, поэтому нормализуем и запрос.
    norm_before = float(np.linalg.norm(query_vector))
    raw_embedding = query_vector[0].astype(float).tolist()
    faiss.normalize_L2(query_vector)
    norm_after = float(np.linalg.norm(query_vector))
    search_started = time.perf_counter()
    scores, vector_ids = index.search(query_vector, min(top_k, index.ntotal))
    search_seconds = time.perf_counter() - search_started

    results: list[dict[str, object]] = []
    for rank, (score, vector_id) in enumerate(zip(scores[0], vector_ids[0]), start=1):
        if vector_id < 0:
            continue
        chunk = dict(chunks[int(vector_id)])
        chunk["rank"] = rank
        chunk["score"] = float(score)
        results.append(chunk)

    trace = {
        "embedding_input": embedding_input,
        "embedding_model": str(info["model"]),
        "embedding_dimensions": int(query_vector.shape[1]),
        "embedding_norm_before": norm_before,
        "embedding_norm_after": norm_after,
        "embedding_seconds": embedding_seconds,
        "faiss_search_seconds": search_seconds,
        "query_embedding": raw_embedding,
        "top_k": min(top_k, index.ntotal),
    }
    return results, trace


def search(query: str, top_k: int = 3) -> list[dict[str, object]]:
    """Преобразовать запрос в вектор и вернуть Top-K ближайших чанков."""

    results, _ = search_with_trace(query, top_k)
    return results


def parse_args() -> argparse.Namespace:
    """Прочитать параметры консольного запуска."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Поисковый запрос на английском языке")
    parser.add_argument("--top-k", type=int, default=3, help="Количество результатов")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=700,
        help="Максимальная длина печатаемого фрагмента",
    )
    return parser.parse_args()


def main() -> None:
    """Выполнить поиск и вывести результаты с источниками и позициями."""

    args = parse_args()
    results = search(args.query, args.top_k)
    print(f"Запрос: {args.query}\n")

    for result in results:
        text = str(result["text"])
        if len(text) > args.max_chars:
            text = text[: args.max_chars].rstrip() + "…"
        print(
            f"{result['rank']}. score={result['score']:.4f} | "
            f"{result['title']} | {result['chunk_id']}\n"
            f"   источник: {result['source']}; "
            f"слова: {result['word_start']}-{result['word_end']}\n"
            f"   {text}\n"
        )


if __name__ == "__main__":
    main()
