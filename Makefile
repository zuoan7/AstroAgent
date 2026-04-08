.PHONY: help install install-dev test test-cov lint format type-check clean run-api run-mcp run-ui start-all

help: ## Show this help message
	@echo "AstroAgent - AI-powered astronomy assistant"
	@echo
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run linters
	flake8 src/ tests/
	isort --check-only src/ tests/
	black --check src/ tests/

format: ## Format code
	isort src/ tests/
	black src/ tests/

type-check: ## Run type checking
	mypy src/

check: lint type-check test ## Run all checks

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

run-api: ## Run API server
	python -m src.api.main

run-mcp: ## Run MCP server
	python -m src.services.mcp_server

run-ui: ## Run Streamlit UI
	streamlit run src/services/streamlit_app.py

start-all: ## Start all services (API, MCP, UI)
	./scripts/start.sh

docker-build: ## Build Docker image
	docker build -t astroagent:latest .

docker-run: ## Run in Docker container
	docker run -p 8000:8000 -p 8001:8001 -p 8501:8501 astroagent:latest

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files