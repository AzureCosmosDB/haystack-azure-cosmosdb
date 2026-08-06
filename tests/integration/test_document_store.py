# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
import os

import pytest
from azure.cosmos import PartitionKey
from haystack.dataclasses import Document

from haystack_azure_cosmosdb import AzureCosmosDBNoSqlDocumentStore
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
        database_name="haystack-testing",
        container_name="haystack_container",
        vector_embedding_policy=get_vector_embedding_policy("float32", 768, "cosine"),
        indexing_policy=get_vector_indexing_policy("quantizedFlat"),
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    )
    # Start from a clean container so leftover data from an interrupted run can't
    # cause duplicate-id failures.
    store.delete_documents([doc.id for doc in store.filter_documents()])
    yield store
    store.delete_documents([doc.id for doc in store.filter_documents()])


class TestDocumentStore:
    def test_write_and_count_documents(self, document_store):
        assert document_store.write_documents([Document(content="some text")]) == 1
        assert document_store.count_documents() == 1

    def test_filter_documents(self, document_store):
        document_store.write_documents(get_documents())
        filters = {"field": "meta.chapter", "operator": "==", "value": "intro"}
        result = document_store.filter_documents(filters=filters)
        assert len(result) == 1
        assert result[0].meta["chapter"] == "intro"

    def test_complex_filter(self, document_store):
        document_store.write_documents(get_documents())
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.chapter", "operator": "==", "value": "intro"},
                {"field": "meta.chapter", "operator": "==", "value": "conclusion"},
            ],
        }
        result = document_store.filter_documents(filters=filters)
        assert {doc.meta["chapter"] for doc in result} == {"intro", "conclusion"}

    def test_delete_documents(self, document_store):
        document_store.write_documents([Document(id="doc-1", content="to delete")])
        document_store.delete_documents(["doc-1"])
        assert document_store.count_documents() == 0
