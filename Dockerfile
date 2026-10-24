# Reproducible environment for the uplift metric-evaluation framework.
FROM python:3.10-slim-bookworm

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /project

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc g++ git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/

CMD ["python", "-m", "src.run_pipeline"]