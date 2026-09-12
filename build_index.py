from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from security import inspect_ingest_document


PROJECT_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = PROJECT_DIR / "knowledge_base"
INDEX_DIR = PROJECT_DIR / "vector_index"
INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
INFO_PATH = INDEX_DIR / "index_info.json"

MODEL_NAME = "nomic-embed-text"
EXPECTED_DIMENSION = 768
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_BATCH_SIZE = 16
EXPECTED_DOCUMENT_COUNT = 33

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 100
MAX_CHUNK_WORDS = 300

WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
DOCUMENT_RE = re.compile(
    r"\A# (?P<title>[^\n]+)\n\nCategory: (?P<category>[^\n]+)\n\n(?P<body>.+?)\s*\Z",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class SourceDocument:
    """Очищенный документ и данные, необходимые для ссылки на источник."""

    path: Path
    title: str
    category: str
    body: str


@dataclass(frozen=True)
class WordRange:
    """Полуинтервал слов исходного документа: начало включено, конец исключён."""

    start: int
    end: int

    @property
    def size(self) -> int:
        """Вернуть количество слов в диапазоне."""

        return self.end - self.start


def count_words(text: str) -> int:
    """Посчитать слова тем же способом, который используется при чанкинге."""

    return len(WORD_RE.findall(text))


def load_documents(
    allow_training_canary: bool = False,
) -> list[SourceDocument]:
    """Загрузить только Markdown-файлы и исключить JSON-словарь замен."""

    paths = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if len(paths) != EXPECTED_DOCUMENT_COUNT:
        raise ValueError(
            f"Ожидалось {EXPECTED_DOCUMENT_COUNT} Markdown-документа, "
            f"найдено: {len(paths)}"
        )

    documents: list[SourceDocument] = []
    for path in paths:
        content = path.read_text(encoding="utf-8")
        match = DOCUMENT_RE.fullmatch(content)
        if not match:
            raise ValueError(f"Неверная структура документа: {path}")
        relative_source = path.relative_to(PROJECT_DIR).as_posix()
        body = match.group("body").strip()
        ingest_event = inspect_ingest_document(
            source=relative_source,
            text=body,
            allow_training_canary=allow_training_canary,
        )
        if ingest_event is not None:
            if ingest_event.action == "ingest_blocked":
                raise ValueError(
                    f"Safety-in заблокировал документ {relative_source}: "
                    f"{', '.join(ingest_event.reasons)}"
                )
            print(
                "Safety-in ingest: разрешено учебное исключение "
                f"{relative_source}; причины: {', '.join(ingest_event.reasons)}"
            )
        documents.append(
            SourceDocument(
                path=path,
                title=match.group("title").strip(),
                category=match.group("category").strip(),
                body=body,
            )
        )
    return documents


def find_word_range(
    body_words: list[str],
    chunk_words: list[str],
    search_from: int,
) -> WordRange:
    """Найти последовательность слов чанка в исходном документе."""

    if not chunk_words:
        raise ValueError("После разбиения получен пустой чанк")

    last_start = len(body_words) - len(chunk_words)
    for start in range(max(0, search_from), last_start + 1):
        if body_words[start : start + len(chunk_words)] == chunk_words:
            return WordRange(start=start, end=start + len(chunk_words))

    # Повторный поиск с начала нужен для редких случаев, когда одинаковая фраза
    # встречается несколько раз, а overlap сдвинул ожидаемую позицию назад.
    for start in range(0, last_start + 1):
        if body_words[start : start + len(chunk_words)] == chunk_words:
            return WordRange(start=start, end=start + len(chunk_words))
    raise ValueError("Не удалось сопоставить чанк с исходным текстом")


def merge_short_ranges(ranges: list[WordRange]) -> list[WordRange]:
    """Присоединить чанки короче 100 слов к соседям, не превышая 300 слов."""

    merged: list[WordRange] = []
    pending: WordRange | None = None

    for current in ranges:
        if current.size < MIN_CHUNK_WORDS:
            pending = (
                current
                if pending is None
                else WordRange(min(pending.start, current.start), max(pending.end, current.end))
            )
            if pending.size >= MIN_CHUNK_WORDS:
                merged.append(pending)
                pending = None
            continue

        if pending is not None:
            combined = WordRange(
                min(pending.start, current.start),
                max(pending.end, current.end),
            )
            if combined.size > MAX_CHUNK_WORDS:
                raise ValueError("Короткий чанк нельзя объединить в пределах 300 слов")
            merged.append(combined)
            pending = None
        else:
            merged.append(current)

    if pending is not None:
        if not merged:
            return [pending]
        combined = WordRange(
            min(merged[-1].start, pending.start),
            max(merged[-1].end, pending.end),
        )
        if combined.size > MAX_CHUNK_WORDS:
            raise ValueError("Последний короткий чанк нельзя объединить с предыдущим")
        merged[-1] = combined

    return merged


def split_document(document: SourceDocument) -> list[dict[str, object]]:
    """Разбить документ и сохранить позиции чанков в исходном тексте."""

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
        chunk_size=CHUNK_SIZE_WORDS,
        chunk_overlap=CHUNK_OVERLAP_WORDS,
        length_function=count_words,
        strip_whitespace=True,
    )
    preliminary_chunks = splitter.split_text(document.body)

    body_matches = list(WORD_RE.finditer(document.body))
    body_words = [match.group(0).casefold() for match in body_matches]
    preliminary_ranges: list[WordRange] = []
    previous_start = -1

    for text in preliminary_chunks:
        chunk_words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
        word_range = find_word_range(body_words, chunk_words, previous_start + 1)
        preliminary_ranges.append(word_range)
        previous_start = word_range.start

    ranges = merge_short_ranges(preliminary_ranges)
    chunks: list[dict[str, object]] = []

    for position, word_range in enumerate(ranges):
        document_is_short = len(body_words) < MIN_CHUNK_WORDS
        size_is_valid = (
            word_range.size <= MAX_CHUNK_WORDS
            and (word_range.size >= MIN_CHUNK_WORDS or document_is_short)
        )
        if not size_is_valid:
            raise ValueError(
                f"{document.path.name}: размер чанка {word_range.size} вне диапазона "
                f"{MIN_CHUNK_WORDS}-{MAX_CHUNK_WORDS}"
            )

        char_start = body_matches[word_range.start].start()
        char_end = body_matches[word_range.end - 1].end()
        while char_end < len(document.body) and document.body[char_end] in ".,;:!?\"'”’)]":
            char_end += 1

        chunks.append(
            {
                "source": document.path.relative_to(PROJECT_DIR).as_posix(),
                "title": document.title,
                "category": document.category,
                "document_chunk": position,
                "word_start": word_range.start,
                "word_end": word_range.end,
                "char_start": char_start,
                "char_end": char_end,
                "word_count": word_range.size,
                "text": document.body[char_start:char_end].strip(),
            }
        )
    return chunks


def build_chunks(documents: list[SourceDocument]) -> list[dict[str, object]]:
    """Сформировать общий список чанков и назначить идентификаторы векторов."""

    chunks = [chunk for document in documents for chunk in split_document(document)]
    for vector_id, chunk in enumerate(chunks):
        chunk["vector_id"] = vector_id
        source_stem = Path(str(chunk["source"])).stem
        chunk["chunk_id"] = f"{source_stem}#{int(chunk['document_chunk']):03d}"
    return chunks


def embedding_text(chunk: dict[str, object]) -> str:
    """Добавить к тексту поисковый префикс и контекст документа."""

    return (
        "search_document: "
        f"Title: {chunk['title']}\n"
        f"Category: {chunk['category']}\n"
        f"{chunk['text']}"
    )


def create_embeddings(chunks: list[dict[str, object]]) -> tuple[np.ndarray, float]:
    """Получить эмбеддинги из Ollama пакетами и измерить чистое время вызовов."""

    # Локальный Ollama не должен использовать системный HTTP_PROXY: иначе даже
    # запрос к localhost может ошибочно уйти во внешний proxy-сервис.
    client = ollama.Client(host=OLLAMA_BASE_URL, trust_env=False)
    vectors: list[list[float]] = []
    started = time.perf_counter()

    for offset in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[offset : offset + EMBEDDING_BATCH_SIZE]
        response = client.embed(
            model=MODEL_NAME,
            input=[embedding_text(chunk) for chunk in batch],
            truncate=False,
            keep_alive="10m",
        )
        vectors.extend(response.embeddings)
        print(f"Эмбеддинги: {min(offset + len(batch), len(chunks))}/{len(chunks)}")

    elapsed = time.perf_counter() - started
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape != (len(chunks), EXPECTED_DIMENSION):
        raise ValueError(
            f"Неожиданная форма матрицы: {matrix.shape}; "
            f"ожидалась ({len(chunks)}, {EXPECTED_DIMENSION})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Эмбеддинги содержат NaN или бесконечные значения")
    return matrix, elapsed


def write_faiss_index(index: faiss.Index, path: Path) -> None:
    """Сохранить FAISS-индекс по пути, который может содержать кириллицу."""

    serialized = faiss.serialize_index(index)
    path.write_bytes(serialized.tobytes())


def read_faiss_index(path: Path) -> faiss.Index:
    """Загрузить FAISS-индекс из Unicode-пути без нативного fopen."""

    serialized = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return faiss.deserialize_index(serialized)


def save_index(
    chunks: list[dict[str, object]],
    embeddings: np.ndarray,
    document_count: int,
    embedding_seconds: float,
    total_seconds: float,
) -> None:
    """Нормализовать векторы и атомарно сохранить индекс и JSON-описания."""

    # После L2-нормализации скалярное произведение равно косинусной близости.
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    if index.ntotal != len(chunks):
        raise ValueError("Количество векторов FAISS не совпало с количеством чанков")

    info = {
        "model": MODEL_NAME,
        "model_reference": "https://ollama.com/library/nomic-embed-text",
        "dimensions": int(embeddings.shape[1]),
        "index_type": "IndexFlatIP",
        "similarity": "cosine (L2-normalized inner product)",
        "document_count": document_count,
        "chunk_count": len(chunks),
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "min_chunk_words": min(int(chunk["word_count"]) for chunk in chunks),
        "max_chunk_words": max(int(chunk["word_count"]) for chunk in chunks),
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "embedding_seconds": round(embedding_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    temporary_index = INDEX_PATH.with_suffix(".index.tmp")
    temporary_chunks = CHUNKS_PATH.with_suffix(".json.tmp")
    temporary_info = INFO_PATH.with_suffix(".json.tmp")

    write_faiss_index(index, temporary_index)
    temporary_chunks.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_info.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    temporary_index.replace(INDEX_PATH)
    temporary_chunks.replace(CHUNKS_PATH)
    temporary_info.replace(INFO_PATH)


def validate_saved_index() -> None:
    """Повторно загрузить артефакты и проверить их согласованность."""

    index = read_faiss_index(INDEX_PATH)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    info = json.loads(INFO_PATH.read_text(encoding="utf-8"))

    if index.d != EXPECTED_DIMENSION:
        raise ValueError(f"В сохранённом индексе неверная размерность: {index.d}")
    if index.ntotal != len(chunks) or index.ntotal != info["chunk_count"]:
        raise ValueError("Количество векторов и метаданных после загрузки не совпало")
    if [chunk["vector_id"] for chunk in chunks] != list(range(len(chunks))):
        raise ValueError("vector_id должны последовательно соответствовать строкам FAISS")
    if any(not math.isfinite(float(chunk["word_count"])) for chunk in chunks):
        raise ValueError("Метаданные содержат некорректный размер чанка")


def main() -> None:
    """Выполнить полный цикл построения и проверки индекса."""

    total_started = time.perf_counter()
    # По условию задания 5 canary обязана попасть в индекс. В проде здесь
    # должно быть false: опасный документ будет остановлен до создания embedding.
    documents = load_documents(allow_training_canary=True)
    chunks = build_chunks(documents)
    print(f"Подготовлено документов: {len(documents)}, чанков: {len(chunks)}")

    embeddings, embedding_seconds = create_embeddings(chunks)
    total_seconds = time.perf_counter() - total_started
    save_index(
        chunks=chunks,
        embeddings=embeddings,
        document_count=len(documents),
        embedding_seconds=embedding_seconds,
        total_seconds=total_seconds,
    )
    validate_saved_index()
    print(
        f"Индекс сохранён: {INDEX_PATH.relative_to(PROJECT_DIR)}; "
        f"векторов: {len(chunks)}; размерность: {embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()
