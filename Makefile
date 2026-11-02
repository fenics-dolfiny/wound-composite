install:
	uv sync

check:
	uv run ruff check .
	uv run mypy .
	uv run bandit -r wound_composite/
	uv run pip-audit --skip-editable

coverage:
	uv run coverage run -m pytest tests
	uv run coverage report -m
	uv run coverage html
	uv run coverage report --format=total > coverage.txt
	@echo "Coverage: $$(cat coverage.txt)%"

lint:
	uv sync
	uv run ruff check --fix .
	uv run ruff format .
	uv run mypy .
	uv run bandit -r wound_composite/
	uv run pip-audit --skip-editable

gitleaks:
	gitleaks detect --source . --verbose
