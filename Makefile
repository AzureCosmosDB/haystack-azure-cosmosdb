.PHONY: help install test integration-test test-all lint format build clean

PYTHON ?= python
SRC = src/haystack_azure_cosmosdb

help:
	@echo "Available targets:"
	@echo "  install           - install the package with test and lint extras (editable)"
	@echo "  test              - run unit tests"
	@echo "  integration-test  - run integration tests (requires a live Cosmos DB for NoSQL account)"
	@echo "  test-all          - run the full test suite"
	@echo "  lint              - run ruff and mypy"
	@echo "  format            - auto-format with black and ruff"
	@echo "  build             - build the sdist and wheel"
	@echo "  clean             - remove build and cache artifacts"

install:
	$(PYTHON) -m pip install -e ".[test,lint,docs]"

test:
	$(PYTHON) -m pytest tests/unit

integration-test:
	$(PYTHON) -m pytest tests/integration

test-all:
	$(PYTHON) -m pytest tests

lint:
	ruff check $(SRC) tests
	black --check $(SRC) tests
	mypy $(SRC)

format:
	black $(SRC) tests
	ruff check --fix $(SRC) tests

build:
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

clean:
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
