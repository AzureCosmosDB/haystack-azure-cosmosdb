# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_azure_cosmosdb.document_store import AzureCosmosDBNoSqlDocumentStore


@component
class AzureCosmosDBNoSqlEmbeddingRetriever:
    """
    Retrieves documents from an `AzureCosmosDBNoSqlDocumentStore` by vector similarity.

    Usage example:
    ```python
    from haystack_azure_cosmosdb import (
        AzureCosmosDBNoSqlDocumentStore,
        AzureCosmosDBNoSqlEmbeddingRetriever,
    )

    store = AzureCosmosDBNoSqlDocumentStore(...)
    retriever = AzureCosmosDBNoSqlEmbeddingRetriever(document_store=store)
    result = retriever.run(query_embedding=[0.1, 0.2, ...])
    ```
    """

    def __init__(
        self,
        *,
        document_store: AzureCosmosDBNoSqlDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ):
        """
        :param document_store: An instance of `AzureCosmosDBNoSqlDocumentStore`.
        :param filters: Filters applied to the retrieved documents.
        :param top_k: Maximum number of documents to return.
        :raises ValueError: If `document_store` is not an `AzureCosmosDBNoSqlDocumentStore`.
        """
        if not isinstance(document_store, AzureCosmosDBNoSqlDocumentStore):
            msg = "document_store must be an instance of AzureCosmosDBNoSqlDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AzureCosmosDBNoSqlEmbeddingRetriever:
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        data["init_parameters"]["document_store"] = AzureCosmosDBNoSqlDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents from the document store by vector similarity.

        :param query_embedding: The embedding of the query.
        :param filters: Filters applied to the retrieved documents. Overrides the value set at init time.
        :param top_k: Maximum number of documents to return. Overrides the value set at init time.
        :returns: A dictionary with a `documents` key containing the retrieved documents.
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
