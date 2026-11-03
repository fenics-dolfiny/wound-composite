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

check: gitleaks bandit mypy

gitleaks:
	gitleaks detect --source . --verbose

bandit: 
	bandit --severity-level medium -r wound_composite test 

install:
	$(PYTHON) -m pip install -e .

mypy: install
	mypy -p wound_composite
	mypy test

test: install
	$(PYTHON) -P -m pytest -n auto
