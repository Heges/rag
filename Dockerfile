FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_BASE_URL=http://ollama:11434

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY build_index.py search_index.py security.py rag_bot.py run_task5_tests.py ./
COPY knowledge_base ./knowledge_base
COPY vector_index ./vector_index

ENTRYPOINT ["python", "-B", "rag_bot.py"]
