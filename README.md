# haystack-azure-cosmosdb

[![PyPI - Version](https://img.shields.io/pypi/v/haystack-azure-cosmosdb.svg)](https://pypi.org/project/haystack-azure-cosmosdb)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/haystack-azure-cosmosdb.svg)](https://pypi.org/project/haystack-azure-cosmosdb)
[![CI](https://github.com/aayush3011/haystack-azure-cosmosdb/actions/workflows/ci.yml/badge.svg)](https://github.com/aayush3011/haystack-azure-cosmosdb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Azure Cosmos DB for NoSQL](https://learn.microsoft.com/azure/cosmos-db/nosql/) integration for
[Haystack](https://haystack.deepset.ai/). It provides a document store and a vector-search retriever
backed by the native Cosmos DB for NoSQL
[vector search](https://learn.microsoft.com/azure/cosmos-db/nosql/vector-search) capabilities.

---

## Table of Contents

- [Installation](#installation)
- [Integrations](#integrations)
- [Usage](#usage)
- [Metadata filtering](#metadata-filtering)
- [Authentication](#authentication)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

## Installation

```bash
pip install haystack-azure-cosmosdb
```

## Integrations

| Integration | Class | Description |
|---|---|---|
| **Document Store** | `AzureCosmosDBNoSqlDocumentStore` | Stores Haystack `Document`s in a Cosmos DB for NoSQL container with vector indexing, full-text indexing, and metadata filtering. |
| **Embedding Retriever** | `AzureCosmosDBNoSqlEmbeddingRetriever` | Retrieves documents by vector similarity using the native `VectorDistance` function. |
| **Full-Text Retriever** | `AzureCosmosDBNoSqlFullTextRetriever` | Retrieves documents by BM25 relevance using `FullTextScore` with `ORDER BY RANK`. |
| **Hybrid Retriever** | `AzureCosmosDBNoSqlHybridRetriever` | Fuses vector and full-text relevance with Reciprocal Rank Fusion (`RRF`), with optional weights. |

A single `AzureCosmosDBNoSqlDocumentStore` powers all three retrieval modes. Full-text and hybrid
retrieval require `full_text_search_enabled=True` and the
[full-text search feature](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/full-text-search)
enabled on your Cosmos DB account.

## Usage

### Create a document store and write documents

```python
from azure.cosmos import PartitionKey
from haystack import Document
from haystack_azure_cosmosdb import AzureCosmosDBNoSqlDocumentStore

vector_embedding_policy = {
    "vectorEmbeddings": [
        {"path": "/embedding", "dataType": "float32", "dimensions": 768, "distanceFunction": "cosine"}
    ]
}
indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
}

# Reads the connection string from AZURE_COSMOS_NOSQL_CONNECTION_STRING by default.
store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
    database_name="haystack_db",
    container_name="haystack_container",
    vector_embedding_policy=vector_embedding_policy,
    indexing_policy=indexing_policy,
    cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    # Set to True to also enable full-text and hybrid retrieval.
    full_text_search_enabled=True,
)

store.write_documents([Document(content="Azure Cosmos DB is a globally distributed database.")])
print(store.count_documents())
```

When `full_text_search_enabled=True`, the store creates the container with a full-text policy on the
`content` field and adds a matching `fullTextIndexes` entry to the indexing policy (unless you supply
your own `full_text_policy`).

### Vector retrieval in a pipeline

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_azure_cosmosdb import AzureCosmosDBNoSqlEmbeddingRetriever

pipeline = Pipeline()
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", AzureCosmosDBNoSqlEmbeddingRetriever(document_store=store))
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

result = pipeline.run({"text_embedder": {"text": "What is Cosmos DB?"}})
print(result["retriever"]["documents"])
```

### Full-text retrieval

```python
from haystack_azure_cosmosdb import AzureCosmosDBNoSqlFullTextRetriever

retriever = AzureCosmosDBNoSqlFullTextRetriever(document_store=store, top_k=5)
result = retriever.run(query_text="globally distributed database")
print(result["documents"])
```

### Hybrid retrieval (vector + full-text)

```python
from haystack_azure_cosmosdb import AzureCosmosDBNoSqlHybridRetriever

retriever = AzureCosmosDBNoSqlHybridRetriever(document_store=store, top_k=5)
result = retriever.run(
    query_embedding=[0.1, 0.2, ...],
    query_text="globally distributed database",
    # Optional [full_text_weight, vector_weight] for weighted RRF:
    weights=[2.0, 1.0],
)
print(result["documents"])
```

Hybrid results are ordered by the fused RRF rank; each document's `score` carries its raw
`VectorDistance` similarity for reference.

## Metadata filtering

The document store supports the standard
[Haystack filter syntax](https://docs.haystack.deepset.ai/docs/metadata-filtering), which is
translated to parameterized Azure Cosmos DB for NoSQL `WHERE` clauses:

```python
filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.chapter", "operator": "==", "value": "intro"},
        {"field": "meta.number", "operator": ">=", "value": 100},
    ],
}

store.filter_documents(filters=filters)
```

Supported comparison operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`.
Supported logical operators: `AND`, `OR`, `NOT`.

## Authentication

The document store supports several authentication methods. Each one reads sensible defaults from
environment variables, so you can also configure it entirely through the environment:

| Variable | Used by | Purpose |
|---|---|---|
| `AZURE_COSMOS_NOSQL_CONNECTION_STRING` | `from_connection_string` | Full account connection string |
| `AZURE_COSMOS_NOSQL_ENDPOINT` | `from_uri_and_key`, `from_aad_token` | Account endpoint URI |
| `AZURE_COSMOS_NOSQL_KEY` | `from_uri_and_key` | Account key |

```python
from azure.cosmos import PartitionKey
from haystack.utils import Secret
from haystack_azure_cosmosdb import AzureCosmosDBNoSqlDocumentStore

common = {
    "database_name": "haystack_db",
    "container_name": "haystack_container",
    "vector_embedding_policy": vector_embedding_policy,
    "indexing_policy": indexing_policy,
    "cosmos_container_properties": {"partition_key": PartitionKey(path="/id")},
}

# 1. Connection string (defaults to env var AZURE_COSMOS_NOSQL_CONNECTION_STRING)
store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(**common)

# 2. Account URI + key (default to env vars AZURE_COSMOS_NOSQL_ENDPOINT and AZURE_COSMOS_NOSQL_KEY);
#    you can also pass them explicitly:
store = AzureCosmosDBNoSqlDocumentStore.from_uri_and_key(
    uri="https://<account>.documents.azure.com:443/",
    key=Secret.from_env_var("AZURE_COSMOS_NOSQL_KEY"),
    **common,
)

# 3. Microsoft Entra ID (AAD / Managed Identity) - endpoint defaults to AZURE_COSMOS_NOSQL_ENDPOINT,
#    credential defaults to DefaultAzureCredential
store = AzureCosmosDBNoSqlDocumentStore.from_aad_token(
    uri="https://<account>.documents.azure.com:443/", **common
)
```

## Examples

See [`examples/retrieval.py`](examples/retrieval.py) for a complete, runnable script that indexes
documents and queries them with all three retrieval modes — vector, full-text, and hybrid — from a
single document store.

## Development

This project uses [Hatch](https://hatch.pypa.io/) for building and a `Makefile` for common tasks.

```bash
make install   # install the package with test and lint extras
make test      # run unit tests
make lint      # run ruff and mypy
make format    # auto-format with black and ruff
```

Integration tests require a live Azure Cosmos DB for NoSQL account. Export the connection string and
run them explicitly:

```bash
export AZURE_COSMOS_NOSQL_CONNECTION_STRING="AccountEndpoint=...;AccountKey=...;"
make integration-test
```

If the connection string is not set, all integration tests are skipped.

## License

`haystack-azure-cosmosdb` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.
