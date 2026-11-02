all: format check

format: typos ruff

typos:
	uv run typos .

ruff:
	uv run ruff format --check
	uv run ruff check

check: gitleaks

gitleaks:
	gitleaks detect --source . --verbose
