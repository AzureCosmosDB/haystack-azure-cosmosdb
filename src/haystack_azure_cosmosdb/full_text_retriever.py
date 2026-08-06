# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_azure_cosmosdb.document_store import AzureCosmosDBNoSqlDocumentStore


@component
class AzureCosmosDBNoSqlFullTextRetriever:
    """
    Retrieves documents from an `AzureCosmosDBNoSqlDocumentStore` using BM25 full-text ranking.

    The document store must be created with `full_text_search_enabled=True`.

    Usage example:
    ```python
    from haystack_azure_cosmosdb import (
        AzureCosmosDBNoSqlDocumentStore,
        AzureCosmosDBNoSqlFullTextRetriever,
    )

    store = AzureCosmosDBNoSqlDocumentStore(..., full_text_search_enabled=True)
    retriever = AzureCosmosDBNoSqlFullTextRetriever(document_store=store)
    result = retriever.run(query_text="azure cosmos db")
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
    def from_dict(cls, data: dict[str, Any]) -> AzureCosmosDBNoSqlFullTextRetriever:
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
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents from the document store using full-text search.

        :param query_text: The free-text query.
        :param filters: Filters applied to the retrieved documents. Overrides the value set at init time.
        :param top_k: Maximum number of documents to return. Overrides the value set at init time.
        :returns: A dictionary with a `documents` key containing the retrieved documents.
        """
        docs = self.document_store._full_text_retrieval(
            query_text=query_text,
            filters=filters or self.filters,
            top_k=top_k or self.top_k,
        )
        return {"documents": docs}
