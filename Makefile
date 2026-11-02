.PHONY: all check gitleaks

all: format check

format: typos

typos:
	uv run typos .

check: gitleaks

gitleaks:
	gitleaks detect --source . --verbose
