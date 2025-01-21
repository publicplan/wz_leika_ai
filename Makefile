BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
COMMIT := $(shell git rev-parse --short HEAD)

.PHONY: help install conda-env conda-env-update \
		data-pull leika wz wz-ci lint requirements

help:
	@echo "install                install package with dependencies."
	@echo "conda-env              create the conda environment 'publicplan-env'."
	@echo "conda-env-update       update 'publicplan-env'."
	@echo "data-pull              download data and model weights. Needs dvc."
	@echo "wz                     create wz-api container."
	@echo "wz-ci                  create wz-api container for ci."
	@echo "leika                  create leika-api container."
	@echo "test                   run unit tests."
	@echo "lint                   run linters."
	@echo "requirements           compile requirements."

install:
	pip install -r requirements/requirements.txt \
		-r requirements/dev-requirements.txt
	pip install -e .

conda-env:
	conda env create --file environment.yml

conda-env-update:
	conda env update --name publicplan-env --file environment.yml

data-pull:
	dvc pull -r readonly-upstream

leika:
	docker build . -f docker/pp-common.Dockerfile \
		--target env-builder -t pp-env-builder
	docker build . -f docker/pp-common.Dockerfile \
		-t pp-common
	docker build . -f docker/leika-api.Dockerfile \
		-t leika-api

wz:
	docker build . -f docker/wz-api.Dockerfile \
		--target env-builder -t pp-env-builder
	docker build . -f docker/wz-api.Dockerfile \
		-t wz-api
	docker tag wz-api:latest publicplan/wz-api-ai:$(subst /,-,$(BRANCH))-$(COMMIT)

wz-ci:
	docker build . -f docker/wz-api.Dockerfile \
		--target env-builder \
		--cache-from registry.gitlab.com/didado/publicplan/pp-env-builder-ci \
		-t registry.gitlab.com/didado/publicplan/pp-env-builder-ci
	docker build . -f docker/wz-api.Dockerfile \
		--cache-from registry.gitlab.com/didado/publicplan/pp-env-builder-ci \
		--cache-from registry.gitlab.com/didado/publicplan/wz-ci \
		-t registry.gitlab.com/didado/publicplan/wz-ci

test:
	pytest -v

lint:
	mypy publicplan tests
	pylint -v publicplan tests
	yapf3 --diff $(shell git ls-files | grep '.py$$')
	test -z "${ dvc status -c | grep new }"

requirements:
	pip-compile -f https://download.pytorch.org/whl/torch_stable.html \
		requirements/requirements.in
	pip-compile requirements/dev-requirements.in
