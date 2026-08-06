# Contributing to haystack-azure-cosmosdb

Thanks for your interest in contributing! This document explains how to set up your environment and
the conventions we follow.

## Development setup

1. Fork and clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```
3. Install the package with development extras:
   ```bash
   make install
   ```

## Workflow

- Create a feature branch prefixed with your username, e.g. `users/<username>/<short-description>`.
- Make your changes with clear, focused commits.
- Ensure linting, type checking, and tests pass before opening a pull request:
  ```bash
  make lint
  make test
  ```
- Use one of the [Conventional Commit](https://www.conventionalcommits.org/) prefixes for your PR
  title: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`, `build:`, `perf:`, `style:`.

## Tests

- **Unit tests** live in `tests/unit/` and run without any external dependencies.
- **Integration tests** live in `tests/integration/` and require a live Azure Cosmos DB for NoSQL
  account. Export the connection string before running them:
  ```bash
  export AZURE_COSMOS_NOSQL_CONNECTION_STRING="AccountEndpoint=...;AccountKey=...;"
  make integration-test
  ```
  Integration tests are automatically skipped when the connection string is not set.

## Code style

- Formatting is handled by [black](https://black.readthedocs.io/) (line length 120).
- Linting is handled by [ruff](https://docs.astral.sh/ruff/).
- Type checking is handled by [mypy](https://mypy-lang.org/).
- Run `make format` to auto-format your changes.

## Reporting issues

Please use the GitHub issue templates to report bugs or request features.
