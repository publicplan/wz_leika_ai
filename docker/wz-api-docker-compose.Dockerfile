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

RUN apt-get update && apt-get install -qq -y git

COPY ./docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
