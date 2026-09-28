# dspy-jev developer commands. `make help` lists them.
SHELL := /usr/bin/env bash
VENV  := .venv
PY    := $(VENV)/bin/python
PYTEST:= $(VENV)/bin/pytest
RUFF  := $(VENV)/bin/ruff

.DEFAULT_GOAL := help
.PHONY: help install dev test test-fast test-slow test-live lint fmt typecheck coverage \
        calibrate evaluate serve mcp mlflow doctor golden clean ci

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install the project with all extras
	./scripts/install.sh service,observability,mcp

dev: install ## Install development dependencies as well
	$(PY) -m pip install -e ".[dev]"

test-fast: ## Unit + contract tests only (no calibration, seconds)
	$(PYTEST) tests/unit tests/contract -q

test: ## Everything except the live provider tests
	$(PYTEST) -m "not live" -q

test-slow: ## Calibration and integration tests
	$(PYTEST) -m "slow and not live" -q

test-live: ## Opt-in tests against real providers (needs OLLAMA_API_KEY / ANTHROPIC_API_KEY)
	$(PYTEST) -m live -q

coverage: ## Test with coverage, enforcing the configured floor
	$(PYTEST) -m "not live" --cov=dspy_jev --cov-report=term-missing --cov-report=xml -q

golden: ## Re-record the golden contract files, then review the diff
	UPDATE_GOLDEN=1 $(PYTEST) tests/contract/test_schema_stability.py -q
	git diff --stat tests/contract/golden

lint: ## Ruff check
	$(RUFF) check src tests

fmt: ## Ruff format and autofix
	$(RUFF) format src tests
	$(RUFF) check --fix src tests

ci: lint coverage ## What CI runs

doctor: ## Verify credentials, models and calibration
	$(VENV)/bin/dspy-jev doctor

calibrate: ## Fit decision parameters on the shipped dataset
	$(VENV)/bin/dspy-jev calibrate --dataset data/action_gate.jsonl

evaluate: ## Score the current gate, failing below the safety floor
	$(VENV)/bin/dspy-jev evaluate --dataset data/action_gate.jsonl --min-safety-recall 0.9

serve: ## Run the HTTP service
	$(VENV)/bin/dspy-jev serve

mcp: ## Run the MCP server over stdio
	$(VENV)/bin/dspy-jev mcp --transport stdio

mlflow: ## Start a local MLflow tracking server
	./scripts/observability.sh

clean: ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
