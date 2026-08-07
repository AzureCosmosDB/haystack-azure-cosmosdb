# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
import os

import pytest
from azure.cosmos import PartitionKey

from haystack_azure_cosmosdb import (
    AzureCosmosDBNoSqlDocumentStore,
    AzureCosmosDBNoSqlFullTextRetriever,
    AzureCosmosDBNoSqlHybridRetriever,
)
from tests.utils import get_documents, get_vector_embedding_policy, get_vector_indexing_policy

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "AZURE_COSMOS_NOSQL_CONNECTION_STRING" not in os.environ,
        reason="No Azure Cosmos DB for NoSQL connection string provided",
    ),
]


@pytest.fixture
def document_store():
    store = AzureCosmosDBNoSqlDocumentStore.from_connection_string(
        database_name="haystack_db",
        container_name="haystack_fts_container",
        vector_embedding_policy=get_vector_embedding_policy("float32", 768, "cosine"),
        indexing_policy=get_vector_indexing_policy("quantizedFlat"),
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        full_text_search_enabled=True,
    )
    store.write_documents(get_documents())
    yield store
    store.delete_documents([doc.id for doc in store.filter_documents()])


class TestFullTextRetrieval:
    def test_full_text_retrieval(self, document_store):
        results = document_store._full_text_retrieval(query_text="Document", top_k=3)
        assert len(results) >= 1

    def test_full_text_retriever_component(self, document_store):
        retriever = AzureCosmosDBNoSqlFullTextRetriever(document_store=document_store, top_k=2)
        result = retriever.run(query_text="Document")
        assert len(result["documents"]) >= 1


class TestHybridRetrieval:
    def test_hybrid_retrieval(self, document_store):
        results = document_store._hybrid_retrieval(query_embedding=[0.1] * 768, query_text="Document", top_k=3)
        assert len(results) >= 1

    def test_hybrid_retrieval_weighted(self, document_store):
        results = document_store._hybrid_retrieval(
            query_embedding=[0.1] * 768, query_text="Document", top_k=3, weights=[2.0, 1.0]
        )
        assert len(results) >= 1

    def test_hybrid_retriever_component(self, document_store):
        retriever = AzureCosmosDBNoSqlHybridRetriever(document_store=document_store, top_k=2)
        result = retriever.run(query_embedding=[0.1] * 768, query_text="Document")
        assert len(result["documents"]) >= 1
