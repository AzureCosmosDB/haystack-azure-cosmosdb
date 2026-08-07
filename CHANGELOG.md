# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-06

### Features Added

- Initial release of `haystack-azure-cosmosdb`.
- `AzureCosmosDBNoSqlDocumentStore` - a Haystack document store backed by Azure Cosmos DB for NoSQL,
  supporting `write_documents` (with duplicate policies), `count_documents`, `delete_documents`,
  `filter_documents`, and three retrieval modes from a single store.
- `AzureCosmosDBNoSqlEmbeddingRetriever` - vector similarity search via the native `VectorDistance`
  function.
- `AzureCosmosDBNoSqlFullTextRetriever` - BM25 full-text search via `FullTextScore` with
  `ORDER BY RANK` (opt-in through `full_text_search_enabled`).
- `AzureCosmosDBNoSqlHybridRetriever` - hybrid search fusing vector and full-text relevance with
  Reciprocal Rank Fusion (`RRF`), with optional per-component weights.
- Standard Haystack metadata filtering translated to parameterized Cosmos DB for NoSQL `WHERE`
  clauses.
- Multiple authentication methods: connection string, account URI + key, and Microsoft Entra ID
  (`DefaultAzureCredential`). Each reads sensible defaults from environment variables -
  `AZURE_COSMOS_NOSQL_CONNECTION_STRING`, `AZURE_COSMOS_NOSQL_ENDPOINT`, and `AZURE_COSMOS_NOSQL_KEY`.
- Full `to_dict` / `from_dict` serialization support for use in Haystack pipelines.
