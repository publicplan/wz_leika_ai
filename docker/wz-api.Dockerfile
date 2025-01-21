FROM python:3.7-slim AS env-builder

RUN apt-get update && apt-get install -qq -y \
  build-essential --no-install-recommends
RUN python -m venv /opt/venv

COPY requirements/requirements.txt ./
RUN . /opt/venv/bin/activate && pip install -r requirements.txt

FROM python:3.7-slim

WORKDIR /app

COPY --from=env-builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY ./.dvc/config /app/.dvc/config
COPY ./data/nlp /app/data/nlp
RUN dvc pull -r readonly-upstream -R data/nlp && \
    rm -r /app/.dvc/cache


COPY ./data/wz /app/data/wz
RUN dvc pull -r readonly-upstream -R data/wz && \
    rm -r /app/.dvc/cache

COPY ./weights/wz /app/weights/wz
RUN dvc pull -r readonly-upstream -R weights/wz && \
    rm -r /app/.dvc/cache

COPY . /app
RUN pip install -e .

ENTRYPOINT ["wz-api", "--host", "0.0.0.0", "--port", "80", "--log-dir",  "/api_log"]
