from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import ollama

from build_index import OLLAMA_BASE_URL
from search_index import search_with_trace
from security import (
    DEFAULT_SECURITY_MODE,
    SECURITY_MODES,
    contains_sensitive_output,
    inspect_user_query,
    protect_chunks,
    redact_structure,
    security_prompt_for,
    validate_security_mode,
)


LLM_MODEL = "llama3.1:8b"
TOP_K = 3
MIN_RELEVANCE_SCORE = 0.65
UNKNOWN_ANSWER = "Я не знаю"
UNKNOWN_PATTERN = re.compile(
    r"^\s*(?:ОТВЕТ:\s*)?Я не знаю[.!]?"
    r"(?:\s+ПРОВЕРЯЕМОЕ ОБОСНОВАНИЕ ПО ИСТОЧНИКАМ:[\s\S]*)?\s*$",
    flags=re.IGNORECASE,
)
CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")
MODEL_OPTIONS = {
    "temperature": 0,
    "seed": 42,
    "num_ctx": 4096,
    "num_predict": 350,
}

SYSTEM_PROMPT = """Ты — RAG-помощник, который отвечает только по переданному контексту.

Правила:
1. Не используй знания, которых нет в контексте, даже если ты знаешь ответ по памяти.
2. Если контекст не содержит прямого ответа, верни только одну строку без заголовка и точки: Я не знаю
3. Для известного ответа используй формат:
ОТВЕТ:
<краткий точный ответ на русском языке>

ПРОВЕРЯЕМОЕ ОБОСНОВАНИЕ ПО ИСТОЧНИКАМ:
1. <какой найденный документ использован>
2. <перескажи найденный факт своими словами полностью по-русски, без английской цитаты>
3. <как этот факт подтверждает ответ>
4. <дополнительный проверяемый факт, только если он нужен>
5. Не выводи номера, пути и список источников: программа добавит их сама без участия модели.
6. Показывай только проверяемое обоснование по тексту. Не выдумывай скрытые мысли модели.
7. Весь ответ и проверяемое обоснование пиши по-русски, даже если вопрос или контекст даны на другом языке.
8. Переводи на русский общие слова и технические описания. Латинские имена собственные и буквенно-цифровые обозначения копируй из контекста дословно: пиши `Vex`, а не `Векс`. Фразу "twin P-s4 ion engines" передавай только как "два ионных двигателя P-s4"; вариант "twin P-s4 ионные двигатели" запрещён.
9. После заголовка ОТВЕТ должна стоять полноценная грамматическая фраза на русском языке, а не скопированный английский фрагмент.
10. В обосновании также пересказывай английские факты по-русски; не вставляй целые английские фразы или цитаты.
"""

QUERY_TRANSLATION_PROMPT = """Переведи запрос пользователя с русского языка на английский для поиска по англоязычной базе знаний.
Не отвечай на вопрос и не добавляй пояснений. Верни только перевод одной строкой.
Сохраняй без перевода вымышленные имена, названия и обозначения, например Vex, Kael Orin, Void Core и P-s4.
Исправляй очевидные опечатки по смыслу запроса."""

FEW_SHOT_MESSAGES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": """КОНТЕКСТ:
[1] Kael Orin
During the Battle of Eryon, Orin saved the Alliance from annihilation by destroying the Dominion's planet-killing superweapon, the Void Core.

ВОПРОС:
Какое супероружие, способное уничтожать планеты, разрушил Kael Orin?""",
    },
    {
        "role": "assistant",
        "content": """ОТВЕТ:
Kael Orin уничтожил Void Core.

ПРОВЕРЯЕМОЕ ОБОСНОВАНИЕ ПО ИСТОЧНИКАМ:
1. Использован документ о Kael Orin.
2. В нём прямо сказано, что Orin уничтожил супероружие, способное уничтожать планеты.
3. В том же предложении это супероружие названо Void Core.""",
    },
    {
        "role": "user",
        "content": """КОНТЕКСТ:
[1] Directive Black
Directive Black identified all Wardens as traitors to the Celestial Commonwealth and therefore subject to summary execution by the Commonwealth Legion.

ВОПРОС:
Против кого была направлена Directive Black?""",
    },
    {
        "role": "assistant",
        "content": """ОТВЕТ:
Directive Black была направлена против Wardens.

ПРОВЕРЯЕМОЕ ОБОСНОВАНИЕ ПО ИСТОЧНИКАМ:
1. Использован документ о Directive Black.
2. В нём все Wardens прямо объявлены предателями.
3. Документ предписывает их немедленную казнь, следовательно, именно они являлись целью.""",
    },
]


@dataclass
class RagResult:
    """Все наблюдаемые данные одного прохода RAG-пайплайна."""

    query: str
    retrieval_query: str
    translation_trace: dict[str, Any] | None
    security_mode: str
    retrieved_chunks: list[dict[str, object]]
    chunks: list[dict[str, object]]
    security_events: list[dict[str, Any]]
    retrieval_trace: dict[str, object]
    input_blocked: bool
    threshold_passed: bool
    messages: list[dict[str, str]]
    raw_response: dict[str, Any] | None
    output_blocked: bool
    answer: str
    total_seconds: float


def build_context(
    chunks: list[dict[str, object]],
    mark_as_untrusted: bool = False,
) -> str:
    """Собрать найденные чанки в размеченный контекст для LLM."""

    sections: list[str] = []
    for number, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[{number}] title={chunk['title']}\n"
            f"text:\n{chunk['text']}"
        )
    context = "\n\n".join(sections)
    if not mark_as_untrusted:
        return context
    return (
        "<<<НАЧАЛО НЕДОВЕРЕННОГО КОНТЕКСТА>>>\n"
        f"{context}\n"
        "<<<КОНЕЦ НЕДОВЕРЕННОГО КОНТЕКСТА>>>"
    )


def build_messages(
    query: str,
    chunks: list[dict[str, object]],
    security_mode: str = DEFAULT_SECURITY_MODE,
) -> list[dict[str, str]]:
    """Сформировать System prompt, два Few-shot примера и текущий запрос."""

    normalized_mode = validate_security_mode(security_mode)
    security_prompt = security_prompt_for(normalized_mode)
    system_prompt = (
        SYSTEM_PROMPT
        if not security_prompt
        else f"{SYSTEM_PROMPT}\n{security_prompt}"
    )
    user_prompt = (
        "КОНТЕКСТ:\n"
        f"{build_context(chunks, mark_as_untrusted=normalized_mode != 'off')}\n\n"
        "ВОПРОС:\n"
        f"{query.strip()}"
    )
    return [
        {"role": "system", "content": system_prompt},
        *FEW_SHOT_MESSAGES,
        {"role": "user", "content": user_prompt},
    ]


def prepare_retrieval_query(query: str) -> tuple[str, dict[str, Any] | None]:
    """Перевести русский вопрос для поиска по англоязычному FAISS-индексу."""

    if CYRILLIC_PATTERN.search(query) is None:
        return query.strip(), None

    started = time.perf_counter()
    client = ollama.Client(host=OLLAMA_BASE_URL, trust_env=False)
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": QUERY_TRANSLATION_PROMPT},
            {"role": "user", "content": query.strip()},
        ],
        options={
            "temperature": 0,
            "seed": 42,
            # Совпадает с основным вызовом, чтобы Ollama не перезагружала runner.
            "num_ctx": 4096,
            "num_predict": 80,
        },
        keep_alive="10m",
    )
    translated_query = response.message.content.strip()
    translated_query = re.sub(
        r"^(?:translation|перевод)\s*:\s*",
        "",
        translated_query,
        flags=re.IGNORECASE,
    ).strip().strip('"')
    if not translated_query:
        raise ValueError("Ollama вернула пустой перевод поискового запроса")

    return translated_query, {
        "translation_applied": True,
        "translation_seconds": time.perf_counter() - started,
        "translation_prompt": QUERY_TRANSLATION_PROMPT,
        "raw_response": response.model_dump(mode="json"),
    }


def ask(
    query: str,
    top_k: int = TOP_K,
    security_mode: str = DEFAULT_SECURITY_MODE,
) -> RagResult:
    """Выполнить поиск, проверить порог и при необходимости вызвать LLM."""

    if not query.strip():
        raise ValueError("Вопрос не должен быть пустым")

    normalized_mode = validate_security_mode(security_mode)
    total_started = time.perf_counter()
    input_event = inspect_user_query(query, normalized_mode)
    if input_event is not None:
        return RagResult(
            query=query,
            retrieval_query="",
            translation_trace=None,
            security_mode=normalized_mode,
            retrieved_chunks=[],
            chunks=[],
            security_events=[input_event.to_dict()],
            retrieval_trace={
                "skipped": True,
                "reason": "safety_in_input_blocked",
            },
            input_blocked=True,
            threshold_passed=False,
            messages=[],
            raw_response=None,
            output_blocked=False,
            answer=UNKNOWN_ANSWER,
            total_seconds=time.perf_counter() - total_started,
        )

    retrieval_query, translation_trace = prepare_retrieval_query(query)
    retrieved_chunks, retrieval_trace = search_with_trace(retrieval_query, top_k)
    chunks, security_event_objects = protect_chunks(
        retrieved_chunks,
        normalized_mode,
    )
    security_events = [event.to_dict() for event in security_event_objects]
    best_score = float(chunks[0]["score"]) if chunks else float("-inf")
    threshold_passed = best_score >= MIN_RELEVANCE_SCORE

    if not threshold_passed:
        return RagResult(
            query=query,
            retrieval_query=retrieval_query,
            translation_trace=translation_trace,
            security_mode=normalized_mode,
            retrieved_chunks=retrieved_chunks,
            chunks=chunks,
            security_events=security_events,
            retrieval_trace=retrieval_trace,
            input_blocked=False,
            threshold_passed=False,
            messages=[],
            raw_response=None,
            output_blocked=False,
            answer=UNKNOWN_ANSWER,
            total_seconds=time.perf_counter() - total_started,
        )

    messages = build_messages(query, chunks, normalized_mode)
    # Локальные запросы не должны перенаправляться через HTTP_PROXY/HTTPS_PROXY.
    client = ollama.Client(host=OLLAMA_BASE_URL, trust_env=False)
    response = client.chat(
        model=LLM_MODEL,
        messages=messages,
        options=MODEL_OPTIONS,
        keep_alive="10m",
    )
    answer = response.message.content.strip()
    if not answer:
        raise ValueError("Ollama вернула пустой ответ")
    # Модель иногда добавляет к отказу заголовок или точку. Нормализуем только
    # заранее разрешённую форму, не переписывая содержательные ответы.
    if is_unknown(answer):
        answer = UNKNOWN_ANSWER

    output_blocked = normalized_mode == "full" and contains_sensitive_output(answer)
    raw_response = response.model_dump(mode="json")
    if output_blocked:
        # Не сохраняем в результате даже сырой вариант заблокированной утечки.
        answer = UNKNOWN_ANSWER
        raw_response = redact_structure(raw_response)

    return RagResult(
        query=query,
        retrieval_query=retrieval_query,
        translation_trace=translation_trace,
        security_mode=normalized_mode,
        retrieved_chunks=retrieved_chunks,
        chunks=chunks,
        security_events=security_events,
        retrieval_trace=retrieval_trace,
        input_blocked=False,
        threshold_passed=True,
        messages=messages,
        raw_response=raw_response,
        output_blocked=output_blocked,
        answer=answer,
        total_seconds=time.perf_counter() - total_started,
    )


def is_unknown(answer: str) -> bool:
    """Определить, отказалась ли модель отвечать из-за отсутствия фактов."""

    return UNKNOWN_PATTERN.fullmatch(answer) is not None


def print_sources(chunks: list[dict[str, object]]) -> None:
    """Напечатать реальные источники программно, без участия LLM."""

    print("\nИСТОЧНИКИ:")
    for number, chunk in enumerate(chunks, start=1):
        print(
            f"[{number}] {chunk['source']} | {chunk['chunk_id']} | "
            f"score={float(chunk['score']):.4f}"
        )


def trace_value(result: RagResult, value: Any) -> Any:
    """Скрыть canary-значение в трассировке режима full."""

    return redact_structure(value) if result.security_mode == "full" else value


def print_trace(result: RagResult) -> None:
    """Показать все данные, которые реально наблюдаемы на уровне программы."""

    print("\n=== ПОЛНАЯ НАБЛЮДАЕМАЯ ТРАССИРОВКА ===")
    print(
        "Примечание: Ollama не возвращает скрытые нейронные рассуждения. "
        "Ниже показаны все доступные входы, вычисления RAG и сырой API-ответ."
    )
    print(f"\n[1] Исходный вопрос:\n{trace_value(result, result.query)}")
    if result.input_blocked:
        print("\nЗапрос не передавался переводчику и модели эмбеддингов.")
    else:
        print(
            "\nЗапрос, переданный модели эмбеддингов:\n"
            f"{trace_value(result, result.retrieval_query)}"
        )
    if result.translation_trace is not None:
        print("\nНаблюдаемая трассировка перевода поискового запроса:")
        print(
            json.dumps(
                trace_value(result, result.translation_trace),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("\nПеревод для поиска не потребовался.")
    print("\n[2] Embedding запроса и измерения:")
    print(
        json.dumps(
            trace_value(result, result.retrieval_trace),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n[3] Полные результаты FAISS:")
    blocked_ids = {
        str(event["chunk_id"])
        for event in result.security_events
        if event["action"] == "dropped"
    }
    for chunk in result.retrieved_chunks:
        chunk_text = str(chunk["text"])
        if result.security_mode == "full" and str(chunk["chunk_id"]) in blocked_ids:
            chunk_text = "[СОДЕРЖИМОЕ СКРЫТО POST-RETRIEVAL ФИЛЬТРОМ]"
        else:
            chunk_text = str(trace_value(result, chunk_text))
        print(
            f"\nrank={chunk['rank']} score={float(chunk['score']):.6f} "
            f"vector_id={chunk['vector_id']} chunk_id={chunk['chunk_id']}\n"
            f"source={chunk['source']} words={chunk['word_start']}-{chunk['word_end']}\n"
            f"{chunk_text}"
        )

    print(
        "\n[3a] Защитный режим и события фильтра:\n"
        f"security_mode={result.security_mode}\n"
        + json.dumps(result.security_events, ensure_ascii=False, indent=2)
    )

    best_score = float(result.chunks[0]["score"]) if result.chunks else float("-inf")
    if result.input_blocked:
        print(
            "\n[4] Решение safety-in:\n"
            "пользовательский запрос заблокирован до перевода, embedding и FAISS; "
            "LLM вызывается=False"
        )
    else:
        print(
            "\n[4] Решение по порогу:\n"
            f"best_score={best_score:.6f}; threshold={MIN_RELEVANCE_SCORE:.2f}; "
            f"LLM вызывается={result.threshold_passed}"
        )

    if result.messages:
        print("\n[5] Точные сообщения, отправленные в LLM:")
        for number, message in enumerate(result.messages, start=1):
            message_content = trace_value(result, message["content"])
            print(
                f"\n--- message {number}; role={message['role']} ---\n"
                f"{message_content}"
            )
        print("\n[6] Параметры генерации:")
        print(json.dumps({"model": LLM_MODEL, **MODEL_OPTIONS}, ensure_ascii=False, indent=2))
        print("\n[7] Сырой ответ Ollama API:")
        print(
            json.dumps(
                trace_value(result, result.raw_response),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        reason = (
            "запрос заблокирован safety-in"
            if result.input_blocked
            else "запрос отклонён порогом релевантности"
        )
        print(f"\n[5-7] LLM для генерации ответа не вызвана: {reason}.")

    print(
        "\n[8] Постобработка:\n"
        f"ответ_unknown={is_unknown(result.answer)}; "
        f"input_blocked={result.input_blocked}; "
        f"safety_out_blocked={result.output_blocked}; "
        f"общее_время={result.total_seconds:.3f} с"
    )
    print("=== КОНЕЦ ТРАССИРОВКИ ===\n")


def print_result(result: RagResult, trace: bool = False) -> None:
    """Вывести трассировку, ответ и только проверенные источники."""

    if trace:
        print_trace(result)
    print(result.answer)
    if not is_unknown(result.answer):
        print_sources(result.chunks)


def parse_args() -> argparse.Namespace:
    """Прочитать параметры однократного запуска или REPL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Один вопрос без запуска REPL")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Количество чанков")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Показать embedding, чанки, промпты, параметры и сырой ответ Ollama",
    )
    parser.add_argument(
        "--security-mode",
        choices=SECURITY_MODES,
        default=DEFAULT_SECURITY_MODE,
        help="Уровень защиты; по умолчанию используется безопасный режим full",
    )
    return parser.parse_args()


def configure_console_encoding() -> None:
    """Использовать UTF-8 даже при перенаправлении вывода в Windows."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_repl(top_k: int, trace: bool, security_mode: str) -> None:
    """Запустить простой интерактивный цикл вопросов и ответов."""

    print(
        f"RAG-бот запущен. Режим защиты: {security_mode}. "
        "Для выхода введите /exit."
    )
    while True:
        try:
            query = input("\nВопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            return
        if query.casefold() == "/exit":
            print("Завершение работы.")
            return
        if not query:
            continue
        try:
            print_result(ask(query, top_k, security_mode), trace)
        except (ValueError, ollama.ResponseError) as error:
            print(f"Ошибка: {error}")


def main() -> None:
    """Выбрать однократный режим или интерактивный REPL."""

    configure_console_encoding()
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k должен быть положительным числом")
    if args.query is not None:
        print_result(
            ask(args.query, args.top_k, args.security_mode),
            args.trace,
        )
    else:
        run_repl(args.top_k, args.trace, args.security_mode)


if __name__ == "__main__":
    main()
