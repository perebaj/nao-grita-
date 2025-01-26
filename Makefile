.PHONY: envinfo
envinfo:
	@poetry env info

## Format code
.PHONY: fmt
fmt:
	poetry run black -l 120 .
	poetry run isort .

## Run linters
.PHONY: lint
lint:
	poetry run flake8 .
	poetry run mypy --ignore-missing-imports .

## Display help for all targets
.PHONY: help
help:
	@awk '/^.PHONY: / { \
		msg = match(lastLine, /^## /); \
			if (msg) { \
				cmd = substr($$0, 9, 100); \
				msg = substr(lastLine, 4, 1000); \
				printf "  ${GREEN}%-30s${RESET} %s\n", cmd, msg; \
			} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST)
