FROM python:3.12-slim

WORKDIR /app
ARG INDOBERT_MODEL_REVISION=main

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Inference uses device=-1. Keep the production versions without downloading
# unused CUDA packages that exhaust the VPS during a rolling build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1+cpu torchvision==0.29.1+cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && python -c "import torch; assert torch.version.cuda is None, 'Expected CPU-only PyTorch'"

# Bake IndoBERT into the image so runtime needs no internet access
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install \
    && git clone --depth 1 https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification \
    /indonesia-bert-sentiment-classification \
    && git -C /indonesia-bert-sentiment-classification checkout "$INDOBERT_MODEL_REVISION" \
    && git -C /indonesia-bert-sentiment-classification lfs pull \
    && python -c "from pathlib import Path; assert Path('/indonesia-bert-sentiment-classification/pytorch_model.bin').stat().st_size > 100_000_000, 'Missing model weights'"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
