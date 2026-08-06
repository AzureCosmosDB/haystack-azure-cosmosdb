# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
import os

import pytest
from azure.cosmos import PartitionKey

from haystack_azure_cosmosdb import AzureCosmosDBNoSqlDocumentStore, AzureCosmosDBNoSqlEmbeddingRetriever
from tests.utils import get_documents, get_vector_embedding_policy, get_vector_indexing_policy

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "AZURE_COSMOS_NOSQL_CONNECTION_STRING" not in os.environ,
        reason="No Azure Cosmos DB for NoSQL connection string provided",
    ),
]


def _store(distance_function: str, index_type: str) -> AzureCosmosDBNoSqlDocumentStore:
    return AzureCosmosDBNoSqlDocumentStore.from_connection_string(
        database_name="haystack_db",
        container_name="haystack_container",
        vector_embedding_policy=get_vector_embedding_policy("float32", 768, distance_function),
        indexing_policy=get_vector_indexing_policy(index_type),
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    )


class TestEmbeddingRetrieval:
    @pytest.mark.parametrize("distance_function", ["cosine", "euclidean", "dotproduct"])
    @pytest.mark.parametrize("index_type", ["flat", "quantizedFlat"])
    def test_embedding_retrieval(self, distance_function, index_type):
        store = _store(distance_function, index_type)
        try:
            store.write_documents(get_documents())
            results = store._embedding_retrieval(query_embedding=[0.1] * 768, top_k=2)
            assert len(results) == 2
            assert results[0].score is not None
        finally:
            store.delete_documents([doc.id for doc in store.filter_documents()])

    def test_retriever_component(self):
        store = _store("cosine", "quantizedFlat")
        try:
            store.write_documents(get_documents())
            retriever = AzureCosmosDBNoSqlEmbeddingRetriever(document_store=store, top_k=2)
            result = retriever.run(query_embedding=[0.1] * 768)
            assert len(result["documents"]) == 2
        finally:
            store.delete_documents([doc.id for doc in store.filter_documents()])
