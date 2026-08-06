# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_azure_cosmosdb.document_store import AzureCosmosDBNoSqlDocumentStore


@component
class AzureCosmosDBNoSqlHybridRetriever:
    """
    Retrieves documents from an `AzureCosmosDBNoSqlDocumentStore` using hybrid search.

    Hybrid search fuses vector similarity and BM25 full-text relevance with Reciprocal Rank Fusion
    (RRF). The document store must be created with `full_text_search_enabled=True`.

    Usage example:
    ```python
    from haystack_azure_cosmosdb import (
        AzureCosmosDBNoSqlDocumentStore,
        AzureCosmosDBNoSqlHybridRetriever,
    )

    store = AzureCosmosDBNoSqlDocumentStore(..., full_text_search_enabled=True)
    retriever = AzureCosmosDBNoSqlHybridRetriever(document_store=store)
    result = retriever.run(query_embedding=[0.1, 0.2, ...], query_text="azure cosmos db")
    ```
    """

    def __init__(
        self,
        *,
        document_store: AzureCosmosDBNoSqlDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        weights: list[float] | None = None,
    ):
        """
        :param document_store: An instance of `AzureCosmosDBNoSqlDocumentStore`.
        :param filters: Filters applied to the retrieved documents.
        :param top_k: Maximum number of documents to return.
        :param weights: Optional per-component RRF weights `[full_text_weight, vector_weight]`.
        :raises ValueError: If `document_store` is not an `AzureCosmosDBNoSqlDocumentStore`.
        """
        if not isinstance(document_store, AzureCosmosDBNoSqlDocumentStore):
            msg = "document_store must be an instance of AzureCosmosDBNoSqlDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.weights = weights

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
            weights=self.weights,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AzureCosmosDBNoSqlHybridRetriever:
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
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        weights: list[float] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents from the document store using hybrid (vector + full-text) search.

        :param query_embedding: The embedding of the query.
        :param query_text: The free-text query.
        :param filters: Filters applied to the retrieved documents. Overrides the value set at init time.
        :param top_k: Maximum number of documents to return. Overrides the value set at init time.
        :param weights: Per-component RRF weights. Overrides the value set at init time.
        :returns: A dictionary with a `documents` key containing the retrieved documents.
        """
        docs = self.document_store._hybrid_retrieval(
            query_embedding=query_embedding,
            query_text=query_text,
            filters=filters or self.filters,
            top_k=top_k or self.top_k,
            weights=weights if weights is not None else self.weights,
        )
        return {"documents": docs}
