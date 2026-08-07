# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
from unittest.mock import MagicMock

import pytest
from azure.cosmos import PartitionKey
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.utils import Secret

from haystack_azure_cosmosdb import (
    AzureCosmosDBNoSqlDocumentStore,
    AzureCosmosDBNoSqlFullTextRetriever,
    AzureCosmosDBNoSqlHybridRetriever,
)
from tests.utils import get_vector_embedding_policy, get_vector_indexing_policy


def _store(*, full_text_search_enabled=True, full_text_policy=None, indexing_policy=None):
    return AzureCosmosDBNoSqlDocumentStore(
        connection_string=Secret.from_env_var("AZURE_COSMOS_NOSQL_CONNECTION_STRING", strict=False),
        database_name="haystack_db",
        container_name="haystack_container",
        vector_embedding_policy=get_vector_embedding_policy(),
        indexing_policy=indexing_policy or get_vector_indexing_policy(),
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        full_text_search_enabled=full_text_search_enabled,
        full_text_policy=full_text_policy,
    )


def _store_with_mock_container(**kwargs):
    store = _store(**kwargs)
    mock = MagicMock()
    mock.query_items.return_value = []
    store._container = mock
    return store, mock


class TestFullTextPolicySetup:
    def test_default_policy_and_index_injected(self):
        store = _store()
        assert store.full_text_policy == {
            "defaultLanguage": "en-US",
            "fullTextPaths": [{"path": "/content", "language": "en-US"}],
        }
        assert store.indexing_policy["fullTextIndexes"] == [{"path": "/content"}]

    def test_default_policy_is_isolated_per_instance(self):
        store_a = _store()
        store_b = _store()
        # Mutating one store's policy must not affect another's (no shared module-level dict).
        store_a.full_text_policy["fullTextPaths"][0]["language"] = "fr-FR"
        assert store_b.full_text_policy["fullTextPaths"][0]["language"] == "en-US"

    def test_disabled_by_default(self):
        store = _store(full_text_search_enabled=False)
        assert store.full_text_search_enabled is False
        assert store.full_text_policy is None
        assert "fullTextIndexes" not in store.indexing_policy

    def test_custom_policy_preserved(self):
        policy = {"defaultLanguage": "en-US", "fullTextPaths": [{"path": "/content", "language": "en-US"}]}
        store = _store(full_text_policy=policy)
        assert store.full_text_policy is policy

    def test_existing_full_text_index_not_overwritten(self):
        indexing = get_vector_indexing_policy()
        indexing["fullTextIndexes"] = [{"path": "/custom"}]
        store = _store(indexing_policy=indexing)
        assert store.indexing_policy["fullTextIndexes"] == [{"path": "/custom"}]

    def test_serialization_round_trip(self):
        store = _store()
        data = store.to_dict()
        assert data["init_parameters"]["full_text_search_enabled"] is True
        assert data["init_parameters"]["full_text_policy"]["fullTextPaths"][0]["path"] == "/content"
        restored = AzureCosmosDBNoSqlDocumentStore.from_dict(data)
        assert restored.full_text_search_enabled is True
        assert restored.full_text_policy == store.full_text_policy
        assert restored.indexing_policy["fullTextIndexes"] == [{"path": "/content"}]


class TestFullTextRetrievalQuery:
    def test_query_and_parameters(self):
        store, mock = _store_with_mock_container()
        store._full_text_retrieval(query_text="azure cosmos db", top_k=5)
        kwargs = mock.query_items.call_args.kwargs
        assert kwargs["query"] == (
            "SELECT TOP @top_k c.id, c.content, c.blob, c.meta, c.embedding FROM c "
            "ORDER BY RANK FullTextScore(c.content, @ft_term_0, @ft_term_1, @ft_term_2)"
        )
        assert kwargs["parameters"] == [
            {"name": "@top_k", "value": 5},
            {"name": "@ft_term_0", "value": "azure"},
            {"name": "@ft_term_1", "value": "cosmos"},
            {"name": "@ft_term_2", "value": "db"},
        ]

    def test_query_with_filters(self):
        store, mock = _store_with_mock_container()
        store._full_text_retrieval(
            query_text="cosmos",
            filters={"field": "meta.chapter", "operator": "==", "value": "intro"},
            top_k=3,
        )
        kwargs = mock.query_items.call_args.kwargs
        assert "WHERE c.meta.chapter = @param0 ORDER BY RANK FullTextScore(c.content, @ft_term_0)" in kwargs["query"]
        assert {"name": "@param0", "value": "intro"} in kwargs["parameters"]

    def test_empty_query_text_raises(self):
        store, _ = _store_with_mock_container()
        with pytest.raises(ValueError, match="query_text must not be empty"):
            store._full_text_retrieval(query_text="   ")

    def test_disabled_raises(self):
        store, _ = _store_with_mock_container(full_text_search_enabled=False)
        with pytest.raises(DocumentStoreError, match="Full-text search is not enabled"):
            store._full_text_retrieval(query_text="cosmos")


class TestHybridRetrievalQuery:
    def test_query_without_weights(self):
        store, mock = _store_with_mock_container()
        store._hybrid_retrieval(query_embedding=[0.1] * 8, query_text="azure db", top_k=4)
        query = mock.query_items.call_args.kwargs["query"]
        assert "VectorDistance(c.embedding, @embedding) AS score FROM c" in query
        assert query.endswith(
            "ORDER BY RANK RRF(FullTextScore(c.content, @ft_term_0, @ft_term_1), "
            "VectorDistance(c.embedding, @embedding))"
        )

    def test_query_with_weights(self):
        store, mock = _store_with_mock_container()
        store._hybrid_retrieval(query_embedding=[0.1] * 8, query_text="azure db", top_k=4, weights=[2.0, 1.0])
        kwargs = mock.query_items.call_args.kwargs
        assert kwargs["query"].endswith(
            "ORDER BY RANK RRF(FullTextScore(c.content, @ft_term_0, @ft_term_1), "
            "VectorDistance(c.embedding, @embedding), [2.0, 1.0])"
        )
        # Weights are embedded as a literal array, not a parameter.
        assert not any(p["name"] == "@weights" for p in kwargs["parameters"])

    def test_weights_wrong_length_raises(self):
        store, _ = _store_with_mock_container()
        with pytest.raises(ValueError, match="exactly two values"):
            store._hybrid_retrieval(query_embedding=[0.1] * 8, query_text="azure db", weights=[1.0, 2.0, 3.0])

    def test_empty_embedding_raises(self):
        store, _ = _store_with_mock_container()
        with pytest.raises(ValueError, match="query_embedding must not be empty"):
            store._hybrid_retrieval(query_embedding=[], query_text="cosmos")

    def test_empty_query_text_raises(self):
        store, _ = _store_with_mock_container()
        with pytest.raises(ValueError, match="query_text must not be empty"):
            store._hybrid_retrieval(query_embedding=[0.1] * 8, query_text="")

    def test_disabled_raises(self):
        store, _ = _store_with_mock_container(full_text_search_enabled=False)
        with pytest.raises(DocumentStoreError, match="Full-text search is not enabled"):
            store._hybrid_retrieval(query_embedding=[0.1] * 8, query_text="cosmos")


class TestFullTextRetriever:
    def test_init_rejects_wrong_store(self):
        with pytest.raises(ValueError, match="AzureCosmosDBNoSqlDocumentStore"):
            AzureCosmosDBNoSqlFullTextRetriever(document_store="not a store")

    def test_to_dict_and_from_dict(self):
        retriever = AzureCosmosDBNoSqlFullTextRetriever(document_store=_store(), top_k=7)
        data = retriever.to_dict()
        assert data["type"].endswith("AzureCosmosDBNoSqlFullTextRetriever")
        assert data["init_parameters"]["top_k"] == 7
        restored = AzureCosmosDBNoSqlFullTextRetriever.from_dict(data)
        assert restored.top_k == 7
        assert isinstance(restored.document_store, AzureCosmosDBNoSqlDocumentStore)

    def test_run_delegates(self, monkeypatch):
        store = _store()
        captured = {}
        monkeypatch.setattr(
            store,
            "_full_text_retrieval",
            lambda query_text, filters, top_k: captured.update(query_text=query_text, filters=filters, top_k=top_k)
            or [Document(content="hit")],
        )
        retriever = AzureCosmosDBNoSqlFullTextRetriever(document_store=store, top_k=3)
        result = retriever.run(query_text="cosmos db")
        assert result == {"documents": [Document(content="hit")]}
        assert captured == {"query_text": "cosmos db", "filters": {}, "top_k": 3}


class TestHybridRetriever:
    def test_init_rejects_wrong_store(self):
        with pytest.raises(ValueError, match="AzureCosmosDBNoSqlDocumentStore"):
            AzureCosmosDBNoSqlHybridRetriever(document_store="not a store")

    def test_to_dict_and_from_dict(self):
        retriever = AzureCosmosDBNoSqlHybridRetriever(document_store=_store(), top_k=7, weights=[2.0, 1.0])
        data = retriever.to_dict()
        assert data["type"].endswith("AzureCosmosDBNoSqlHybridRetriever")
        assert data["init_parameters"]["weights"] == [2.0, 1.0]
        restored = AzureCosmosDBNoSqlHybridRetriever.from_dict(data)
        assert restored.weights == [2.0, 1.0]
        assert isinstance(restored.document_store, AzureCosmosDBNoSqlDocumentStore)

    def test_run_delegates(self, monkeypatch):
        store = _store()
        captured = {}

        def fake(query_embedding, query_text, filters, top_k, weights):
            captured.update(
                query_embedding=query_embedding, query_text=query_text, filters=filters, top_k=top_k, weights=weights
            )
            return [Document(content="hit")]

        monkeypatch.setattr(store, "_hybrid_retrieval", fake)
        retriever = AzureCosmosDBNoSqlHybridRetriever(document_store=store, top_k=3, weights=[1.0, 1.0])
        result = retriever.run(query_embedding=[0.1], query_text="cosmos")
        assert result == {"documents": [Document(content="hit")]}
        assert captured == {
            "query_embedding": [0.1],
            "query_text": "cosmos",
            "filters": {},
            "top_k": 3,
            "weights": [1.0, 1.0],
        }
