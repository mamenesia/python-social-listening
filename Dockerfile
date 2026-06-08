FROM python:3.12-slim

WORKDIR /app
ARG INDOBERT_MODEL_REVISION=main

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake IndoBERT into the image so runtime needs no internet access
RUN git clone --depth 1 https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification \
    /indonesia-bert-sentiment-classification \
    && git -C /indonesia-bert-sentiment-classification checkout "$INDOBERT_MODEL_REVISION"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
