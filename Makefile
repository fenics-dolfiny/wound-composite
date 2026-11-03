.PHONY: install test

PYTHON ?= python

all: format check test

format: prettier typos ruff

prettier:
	npx prettier . --check --plugin prettier-plugin-toml

typos:
	uv run typos .

ruff:
	uv run ruff format --check
	uv run ruff check

check: gitleaks

gitleaks:
	gitleaks detect --source . --verbose

install:
	$(PYTHON) -m pip install -e .

test: install
	$(PYTHON) -P -m pytest -n auto
