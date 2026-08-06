# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
#
# Retrieval example for the Azure Cosmos DB for NoSQL Haystack integration.
#
# A single `AzureCosmosDBNoSqlDocumentStore` powers all three retrieval modes:
#   1. Vector search      -> AzureCosmosDBNoSqlEmbeddingRetriever  (VectorDistance)
#   2. Full-text search   -> AzureCosmosDBNoSqlFullTextRetriever   (FullTextScore / BM25)
#   3. Hybrid search      -> AzureCosmosDBNoSqlHybridRetriever      (RRF over vector + full-text)
#
# Install the required packages:
#
#   pip install haystack-azure-cosmosdb
#
# Prerequisites:
#   - An Azure Cosmos DB for NoSQL account with the Vector Search and Full-Text Search
#     features enabled: https://learn.microsoft.com/azure/cosmos-db/gen-ai/full-text-search
#   - Authentication configured through one of the supported environment variables, e.g.
#       export AZURE_COSMOS_NOSQL_CONNECTION_STRING="AccountEndpoint=...;AccountKey=...;"
#   - An Azure OpenAI embedding deployment for generating embeddings:
#       export AZURE_OPENAI_API_KEY="..."
#       export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
#       export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
#
# Run it:
#
#   python examples/retrieval.py

import os

from azure.cosmos import PartitionKey
from haystack import Document, Pipeline
from haystack.components.embedders import AzureOpenAIDocumentEmbedder, AzureOpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_azure_cosmosdb import (
    AzureCosmosDBNoSqlDocumentStore,
    AzureCosmosDBNoSqlEmbeddingRetriever,
    AzureCosmosDBNoSqlFullTextRetriever,
    AzureCosmosDBNoSqlHybridRetriever,
)

# `text-embedding-3-small` produces 1536-dimensional vectors. Adjust to match your deployment.
EMBEDDING_DIMENSIONS = 1536
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
AZURE_COSMOSDB_DOCUMENT_STORE_URI=os.environ.get("AZURE_COSMOSDB_DOCUMENT_STORE_URI", "")
AZURE_OPENAI_ENDPOINT=os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY=os.environ.get("AZURE_OPENAI_API_KEY", "")


def text_embedder() -> AzureOpenAITextEmbedder:
    return AzureOpenAITextEmbedder(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_key=Secret.from_token(AZURE_OPENAI_API_KEY),
    )


def show(title, docs):
    print(f"=== {title} ===")
    for doc in docs:
        score = f"{doc.score:.4f}" if doc.score is not None else "n/a"
        print(f"  [{score}] {doc.content}")
    print()


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Create the document store (vector + full-text enabled).
    # -----------------------------------------------------------------------
    document_store = AzureCosmosDBNoSqlDocumentStore.from_aad_token(
        uri=AZURE_COSMOSDB_DOCUMENT_STORE_URI,
        database_name="haystack_db",
        container_name="haystack_container",
        vector_embedding_policy={
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "dimensions": EMBEDDING_DIMENSIONS,
                    "distanceFunction": "cosine",
                }
            ]
        },
        indexing_policy={
            "indexingMode": "consistent",
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": '/"_etag"/?'}],
            "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
        },
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        # Enables the full-text and hybrid retrievers below.
        full_text_search_enabled=True,
    )

    # -----------------------------------------------------------------------
    # 2. Index a few documents (embed their content, then write them to Cosmos DB).
    # -----------------------------------------------------------------------
    documents = [
        Document(content="Azure Cosmos DB is a globally distributed, multi-model database service."),
        Document(content="Vector search finds semantically similar content using embeddings."),
        Document(content="Full-text search ranks documents by keyword relevance using BM25 scoring."),
        Document(content="Hybrid search combines vector and full-text relevance with Reciprocal Rank Fusion."),
        Document(content="Haystack is an open-source framework for building LLM-powered applications."),
    ]

    print("Indexing documents...")
    indexing = Pipeline()
    indexing.add_component(
        "embedder",
        AzureOpenAIDocumentEmbedder(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=Secret.from_token(AZURE_OPENAI_API_KEY),
        ),
    )
    indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))
    indexing.connect("embedder", "writer")
    indexing.run({"embedder": {"documents": documents}})
    print(f"Indexed {document_store.count_documents()} documents.\n")

    query = "How does hybrid search work in Cosmos DB?"

    # -----------------------------------------------------------------------
    # 3a. Vector retrieval — embed the query, then search by vector similarity.
    # -----------------------------------------------------------------------
    vector_pipeline = Pipeline()
    vector_pipeline.add_component("text_embedder", text_embedder())
    vector_pipeline.add_component(
        "retriever", AzureCosmosDBNoSqlEmbeddingRetriever(document_store=document_store, top_k=3)
    )
    vector_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    vector_result = vector_pipeline.run({"text_embedder": {"text": query}})
    show("Vector retrieval", vector_result["retriever"]["documents"])

    # -----------------------------------------------------------------------
    # 3b. Full-text retrieval — BM25 keyword ranking, no embedding needed.
    # -----------------------------------------------------------------------
    full_text_retriever = AzureCosmosDBNoSqlFullTextRetriever(document_store=document_store, top_k=3)
    full_text_result = full_text_retriever.run(query_text=query)
    show("Full-text retrieval", full_text_result["documents"])

    # -----------------------------------------------------------------------
    # 3c. Hybrid retrieval — fuse vector + full-text with RRF in a single query.
    #     Pass optional weights as [full_text_weight, vector_weight].
    # -----------------------------------------------------------------------
    hybrid_pipeline = Pipeline()
    hybrid_pipeline.add_component("text_embedder", text_embedder())
    hybrid_pipeline.add_component(
        "retriever", AzureCosmosDBNoSqlHybridRetriever(document_store=document_store, top_k=3)
    )
    hybrid_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    hybrid_result = hybrid_pipeline.run(
        {
            "text_embedder": {"text": query},
            "retriever": {"query_text": query, "weights": [1.0, 1.0]},
        }
    )
    show("Hybrid retrieval (vector + full-text via RRF)", hybrid_result["retriever"]["documents"])


if __name__ == "__main__":
    main()
