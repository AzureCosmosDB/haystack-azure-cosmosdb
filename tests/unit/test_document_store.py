# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
import os
from unittest.mock import patch

import pytest
from azure.cosmos import PartitionKey
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.utils import Secret

from haystack_azure_cosmosdb import AzureCosmosDBNoSqlDocumentStore
from tests.utils import get_vector_embedding_policy, get_vector_indexing_policy


def _store(**overrides):
    params = {
        "connection_string": Secret.from_env_var("AZURE_COSMOS_NOSQL_CONNECTION_STRING", strict=False),
        "database_name": "haystack_db",
        "container_name": "haystack_container",
        "vector_embedding_policy": get_vector_embedding_policy(),
        "indexing_policy": get_vector_indexing_policy(),
        "cosmos_container_properties": {"partition_key": PartitionKey(path="/id")},
    }
    params.update(overrides)
    return AzureCosmosDBNoSqlDocumentStore(**params)


class TestInitValidation:
    def test_missing_vector_indexes_raises(self):
        with pytest.raises(ValueError, match="vectorIndexes"):
            _store(indexing_policy={"indexingMode": "consistent", "vectorIndexes": []})

    def test_missing_vector_embeddings_raises(self):
        with pytest.raises(ValueError, match="vectorEmbeddings"):
            _store(vector_embedding_policy={"vectorEmbeddings": []})

    def test_missing_partition_key_raises(self):
        with pytest.raises(ValueError, match="partition_key"):
            _store(cosmos_container_properties={})

    def test_partition_key_from_string(self):
        store = _store(cosmos_container_properties={"partition_key": "/id"})
        assert store._partition_key_path == "/id"

    def test_partition_key_from_object(self):
        store = _store(cosmos_container_properties={"partition_key": PartitionKey(path="/id")})
        assert store._partition_key_path == "/id"


class TestSerialization:
    def test_to_dict(self):
        store = _store()
        data = store.to_dict()
        assert data["type"] == "haystack_azure_cosmosdb.document_store.AzureCosmosDBNoSqlDocumentStore"
        init = data["init_parameters"]
        assert init["database_name"] == "haystack_db"
        assert init["container_name"] == "haystack_container"
        assert init["cosmos_container_properties"] == {"partition_key": "/id"}
        assert init["connection_string"] == {
            "type": "env_var",
            "env_vars": ["AZURE_COSMOS_NOSQL_CONNECTION_STRING"],
            "strict": False,
        }
        assert init["vector_embedding_policy"]["vectorEmbeddings"][0]["dimensions"] == 768

    def test_from_dict_round_trip(self):
        store = _store()
        restored = AzureCosmosDBNoSqlDocumentStore.from_dict(store.to_dict())
        assert restored.database_name == store.database_name
        assert restored.container_name == store.container_name
        assert restored.indexing_policy == store.indexing_policy
        assert restored.vector_embedding_policy == store.vector_embedding_policy
        assert restored._partition_key_path == "/id"
        assert isinstance(restored.cosmos_container_properties["partition_key"], PartitionKey)


class TestEmbeddingRetrievalValidation:
    def test_empty_query_embedding_raises_without_db(self):
        store = _store()
        with pytest.raises(ValueError, match="must not be empty"):
            store._embedding_retrieval(query_embedding=[])


class TestWriteDocumentsValidation:
    def test_rejects_non_document_items_in_mixed_list(self):
        store = _store()
        with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
            store.write_documents([Document(content="ok"), "bad"])

    def test_empty_list_is_allowed(self):
        store = _store()
        assert store.write_documents([]) == 0


class TestDeleteDocumentsValidation:
    def test_delete_requires_id_partition_key(self):
        store = _store(cosmos_container_properties={"partition_key": PartitionKey(path="/meta_key")})
        with pytest.raises(DocumentStoreError, match="partitioned by '/id'"):
            store.delete_documents(["doc-1"])

    def test_delete_empty_list_is_noop(self):
        # Should return without touching the (uninitialized) container, even for a non-/id key.
        store = _store(cosmos_container_properties={"partition_key": PartitionKey(path="/meta_key")})
        assert store.delete_documents([]) is None


_COMMON = {
    "database_name": "haystack_db",
    "container_name": "haystack_container",
    "vector_embedding_policy": get_vector_embedding_policy(),
    "indexing_policy": get_vector_indexing_policy(),
    "cosmos_container_properties": {"partition_key": PartitionKey(path="/id")},
}


class TestUriAndKeyAuth:
    def test_explicit_string_uri_and_key(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_uri_and_key(
            uri="https://acct.documents.azure.com:443/", key=Secret.from_token("secret-key"), **_COMMON
        )
        assert store.connection_string.resolve_value() == (
            "AccountEndpoint=https://acct.documents.azure.com:443/;AccountKey=secret-key;"
        )

    def test_uri_and_key_default_to_env_vars(self):
        with patch.dict(
            os.environ,
            {
                "AZURE_COSMOS_NOSQL_ENDPOINT": "https://envacct.documents.azure.com:443/",
                "AZURE_COSMOS_NOSQL_KEY": "env-key",
            },
        ):
            store = AzureCosmosDBNoSqlDocumentStore.from_uri_and_key(**_COMMON)
        assert store.connection_string.resolve_value() == (
            "AccountEndpoint=https://envacct.documents.azure.com:443/;AccountKey=env-key;"
        )

    def test_uri_as_secret(self):
        store = AzureCosmosDBNoSqlDocumentStore.from_uri_and_key(
            uri=Secret.from_token("https://secreturi/"), key=Secret.from_token("k"), **_COMMON
        )
        assert store.connection_string.resolve_value() == "AccountEndpoint=https://secreturi/;AccountKey=k;"

    def test_missing_env_vars_raises_on_resolve(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="AZURE_COSMOS_NOSQL_ENDPOINT"):
                AzureCosmosDBNoSqlDocumentStore.from_uri_and_key(**_COMMON)


class TestAadTokenAuth:
    def test_explicit_string_uri(self):
        with (
            patch("haystack_azure_cosmosdb.document_store.CosmosClient") as mock_client,
            patch("haystack_azure_cosmosdb.document_store.DefaultAzureCredential"),
        ):
            store = AzureCosmosDBNoSqlDocumentStore.from_aad_token(
                uri="https://acct.documents.azure.com:443/", **_COMMON
            )
            _ = store.client
        assert mock_client.call_args.args[0] == "https://acct.documents.azure.com:443/"

    def test_uri_defaults_to_env_var(self):
        with (
            patch.dict(os.environ, {"AZURE_COSMOS_NOSQL_ENDPOINT": "https://envacct.documents.azure.com:443/"}),
            patch("haystack_azure_cosmosdb.document_store.CosmosClient") as mock_client,
            patch("haystack_azure_cosmosdb.document_store.DefaultAzureCredential"),
        ):
            store = AzureCosmosDBNoSqlDocumentStore.from_aad_token(**_COMMON)
            _ = store.client
        assert mock_client.call_args.args[0] == "https://envacct.documents.azure.com:443/"
