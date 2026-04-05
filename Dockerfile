FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TOKENIZERS_PARALLELISM=false
ENV MODEL_NAME=sergeyzh/rubert-mini-frida
ENV MODEL_MAX_LENGTH=512
ENV MODEL_DEVICE=cpu
ENV MODEL_DEFAULT_PREFIX="categorize: "

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
