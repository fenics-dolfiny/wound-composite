all: format check

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
