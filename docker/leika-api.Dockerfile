FROM python:3.7-slim AS env-builder

RUN apt-get update && apt-get install -qq -y \
  build-essential --no-install-recommends
RUN python -m venv /opt/venv

COPY requirements/*requirements.txt ./
RUN . /opt/venv/bin/activate && \
  pip install -r requirements.txt

FROM python:3.7-slim

WORKDIR /app

RUN pip install dvc==0.80.0

COPY ./.dvc/config /app/.dvc/config
COPY ./weights/embeddings/fasttext_german*.zip.dvc /app/weights/embeddings/

ARG embedding=fasttext_german
RUN dvc pull -r readonly-upstream "weights/embeddings/${embedding}.zip.dvc" && \
    rm -r /app/.dvc/cache

COPY ./data/nlp /app/data/nlp
RUN dvc pull -r readonly-upstream -R data/nlp && \
    rm -r /app/.dvc/cache

COPY --from=env-builder /opt/venv /opt/venv

COPY ./data/leika /app/data/leika
RUN dvc pull -r readonly-upstream -R data/leika && \
    rm -r /app/.dvc/cache

COPY ./weights/leika /app/weights/leika
RUN dvc pull -r readonly-upstream -R weights/leika && \
    rm -r /app/.dvc/cache

ENV PATH="/opt/venv/bin:$PATH" 
COPY . /app
RUN pip install -e .

ENTRYPOINT ["leika-api", "--host", "0.0.0.0", "--port", "80"]
