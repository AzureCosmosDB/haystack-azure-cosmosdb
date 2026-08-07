# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
import pytest
from azure.cosmos import PartitionKey
from haystack import Document
from haystack.utils import Secret

from haystack_azure_cosmosdb import AzureCosmosDBNoSqlDocumentStore, AzureCosmosDBNoSqlEmbeddingRetriever
from tests.utils import get_vector_embedding_policy, get_vector_indexing_policy


def _store():
    return AzureCosmosDBNoSqlDocumentStore(
        connection_string=Secret.from_env_var("AZURE_COSMOS_NOSQL_CONNECTION_STRING", strict=False),
        database_name="haystack_db",
        container_name="haystack_container",
        vector_embedding_policy=get_vector_embedding_policy(),
        indexing_policy=get_vector_indexing_policy(),
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    )


class TestRetriever:
    def test_init_rejects_wrong_store(self):
        with pytest.raises(ValueError, match="AzureCosmosDBNoSqlDocumentStore"):
            AzureCosmosDBNoSqlEmbeddingRetriever(document_store="not a store")

    def test_init_defaults(self):
        retriever = AzureCosmosDBNoSqlEmbeddingRetriever(document_store=_store())
        assert retriever.filters == {}
        assert retriever.top_k == 10

    def test_to_dict(self):
        retriever = AzureCosmosDBNoSqlEmbeddingRetriever(document_store=_store(), filters={"field": "meta.x"}, top_k=5)
        data = retriever.to_dict()
        assert data["type"] == ("haystack_azure_cosmosdb.embedding_retriever.AzureCosmosDBNoSqlEmbeddingRetriever")
        assert data["init_parameters"]["top_k"] == 5
        assert data["init_parameters"]["filters"] == {"field": "meta.x"}
        assert data["init_parameters"]["document_store"]["init_parameters"]["database_name"] == "haystack_db"

    def test_from_dict_round_trip(self):
        retriever = AzureCosmosDBNoSqlEmbeddingRetriever(document_store=_store(), top_k=7)
        restored = AzureCosmosDBNoSqlEmbeddingRetriever.from_dict(retriever.to_dict())
        assert restored.top_k == 7
        assert isinstance(restored.document_store, AzureCosmosDBNoSqlDocumentStore)
        assert restored.document_store.database_name == "haystack_db"

    def test_run_delegates_to_store(self, monkeypatch):
        store = _store()
        captured = {}

        def fake_retrieval(query_embedding, filters, top_k):
            captured["query_embedding"] = query_embedding
            captured["filters"] = filters
            captured["top_k"] = top_k
            return [Document(content="hit")]

        monkeypatch.setattr(store, "_embedding_retrieval", fake_retrieval)
        retriever = AzureCosmosDBNoSqlEmbeddingRetriever(document_store=store, top_k=3)
        result = retriever.run(query_embedding=[0.1, 0.2])

        assert result == {"documents": [Document(content="hit")]}
        assert captured == {"query_embedding": [0.1, 0.2], "filters": {}, "top_k": 3}

    def test_run_overrides(self, monkeypatch):
        store = _store()
        captured = {}

        def fake_retrieval(query_embedding, filters, top_k):
            captured["query_embedding"] = query_embedding
            captured["filters"] = filters
            captured["top_k"] = top_k
            return []

        monkeypatch.setattr(store, "_embedding_retrieval", fake_retrieval)
        retriever = AzureCosmosDBNoSqlEmbeddingRetriever(document_store=store, top_k=3)
        retriever.run(query_embedding=[0.1], filters={"field": "meta.y"}, top_k=1)
        assert captured == {"query_embedding": [0.1], "filters": {"field": "meta.y"}, "top_k": 1}

    def test_run_can_override_with_empty_filters_and_zero_top_k(self, monkeypatch):
        # filters={} and top_k=0 must override init-time defaults (not fall back via `or`).
        store = _store()
        captured = {}

        def fake_retrieval(query_embedding, filters, top_k):
            captured["query_embedding"] = query_embedding
            captured["filters"] = filters
            captured["top_k"] = top_k
            return []

        monkeypatch.setattr(store, "_embedding_retrieval", fake_retrieval)
        retriever = AzureCosmosDBNoSqlEmbeddingRetriever(document_store=store, filters={"field": "meta.x"}, top_k=10)
        retriever.run(query_embedding=[0.1], filters={}, top_k=0)
        assert captured == {"query_embedding": [0.1], "filters": {}, "top_k": 0}
