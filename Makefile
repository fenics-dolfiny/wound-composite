.PHONY: all check gitleaks

all: check

check: gitleaks

gitleaks:
	gitleaks detect --source . --verbose
