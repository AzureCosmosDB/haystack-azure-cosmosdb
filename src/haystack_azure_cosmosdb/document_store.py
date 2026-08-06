# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from typing import Any, cast

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.container import ContainerProxy
from azure.cosmos.exceptions import CosmosResourceExistsError
from azure.identity import DefaultAzureCredential
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_azure_cosmosdb.filters import _normalize_filters

logger = logging.getLogger(__name__)

USER_AGENT = "HaystackCDBNoSqlDocumentStore"

# Environment variables used as defaults for the various authentication paths.
AZURE_COSMOS_NOSQL_CONNECTION_STRING = "AZURE_COSMOS_NOSQL_CONNECTION_STRING"
AZURE_COSMOS_NOSQL_ENDPOINT = "AZURE_COSMOS_NOSQL_ENDPOINT"
AZURE_COSMOS_NOSQL_KEY = "AZURE_COSMOS_NOSQL_KEY"

# Full-text search operates on the Haystack ``content`` field.
FULL_TEXT_FIELD = "content"
FULL_TEXT_PATH = f"/{FULL_TEXT_FIELD}"
DEFAULT_FULL_TEXT_POLICY = {
    "defaultLanguage": "en-US",
    "fullTextPaths": [{"path": FULL_TEXT_PATH, "language": "en-US"}],
}

# Hybrid RRF fuses two components: full-text score and vector distance.
_HYBRID_RRF_COMPONENTS = 2


class AzureCosmosDBNoSqlDocumentStore:
    """
    A DocumentStore implementation backed by
    [Azure Cosmos DB for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search).

    The document store persists Haystack `Document`s as items in a Cosmos DB container and supports
    three retrieval modes, all from a single store:

    - **Vector** similarity search through the native `VectorDistance` function.
    - **Full-text** (BM25) search through `FullTextScore` with `ORDER BY RANK` (opt-in via
      `full_text_search_enabled`).
    - **Hybrid** search that fuses vector and full-text results with Reciprocal Rank Fusion
      (`RRF`), with optional per-component weights.

    It also supports metadata filtering using the standard Haystack filter syntax.

    The store can be created either by passing an existing `CosmosClient` (via the `cosmos_client`
    argument or one of the `from_*` class methods) or, for a serialization-friendly setup, by
    providing a `connection_string` secret. When neither a client nor a connection string is
    available, the store falls back to `DefaultAzureCredential` (AAD / Managed Identity) using the
    `AZURE_COSMOS_NOSQL_ENDPOINT` environment variable.

    Usage example:
    ```python
    from azure.cosmos import PartitionKey
    from haystack_azure_cosmosdb import AzureCosmosDBNoSqlDocumentStore

    store = AzureCosmosDBNoSqlDocumentStore(
        database_name="haystack_db",
        container_name="haystack_container",
        vector_embedding_policy={
            "vectorEmbeddings": [
                {"path": "/embedding", "dataType": "float32", "dimensions": 768, "distanceFunction": "cosine"}
            ]
        },
        indexing_policy={
            "indexingMode": "consistent",
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": '/"_etag"/?'}],
            "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}],
        },
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    )
    print(store.count_documents())
    ```
    """

    def __init__(
        self,
        *,
        database_name: str,
        container_name: str,
        vector_embedding_policy: dict[str, Any],
        indexing_policy: dict[str, Any],
        cosmos_container_properties: dict[str, Any],
        cosmos_database_properties: dict[str, Any] | None = None,
        full_text_search_enabled: bool = False,
        full_text_policy: dict[str, Any] | None = None,
        connection_string: Secret | None = None,
        cosmos_client: CosmosClient | None = None,
    ):
        """
        Creates a new `AzureCosmosDBNoSqlDocumentStore`.

        :param database_name: Name of the Cosmos DB database to use. Created automatically if missing.
        :param container_name: Name of the container to use. Created automatically if missing.
        :param vector_embedding_policy: The
            [vector embedding policy](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search#container-vector-policies)
            describing the embedding path, data type, dimensions and distance function.
        :param indexing_policy: The indexing policy. It must contain a non-empty `vectorIndexes` entry.
        :param cosmos_container_properties: Container properties. Must contain a `partition_key` entry,
            which can be an `azure.cosmos.PartitionKey` or a path string such as `"/id"`.
        :param cosmos_database_properties: Optional database-level properties (e.g. throughput).
        :param full_text_search_enabled: Enables full-text and hybrid retrieval. When `True`, the
            container is created with a full-text policy on the `content` field and a matching
            full-text index is added to the indexing policy (unless already provided). Requires the
            full-text search feature to be enabled on the Cosmos DB account.
        :param full_text_policy: An explicit
            [full-text policy](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/full-text-search).
            When omitted and `full_text_search_enabled` is `True`, a default policy targeting the
            `content` field is used.
        :param connection_string: A Cosmos DB connection string wrapped in a `Secret`. Used when no
            `cosmos_client` is supplied. Defaults to the `AZURE_COSMOS_NOSQL_CONNECTION_STRING`
            environment variable. This is the serialization-friendly authentication path.
        :param cosmos_client: An already-configured `CosmosClient`. When supplied it takes precedence
            over `connection_string`. Note that a store built from a raw client cannot be serialized
            back to a live client and will re-authenticate from `connection_string` / AAD on load.

        :raises ValueError: If `vectorIndexes` or `vectorEmbeddings` are missing or empty.
        """
        if not indexing_policy.get("vectorIndexes"):
            msg = "vectorIndexes cannot be null or empty in the indexing_policy."
            raise ValueError(msg)
        if not vector_embedding_policy or not vector_embedding_policy.get("vectorEmbeddings"):
            msg = "vectorEmbeddings cannot be null or empty in the vector_embedding_policy."
            raise ValueError(msg)
        if "partition_key" not in cosmos_container_properties:
            msg = "cosmos_container_properties must contain a 'partition_key' entry."
            raise ValueError(msg)

        if connection_string is None:
            connection_string = Secret.from_env_var(AZURE_COSMOS_NOSQL_CONNECTION_STRING, strict=False)

        self.database_name = database_name
        self.container_name = container_name
        self.vector_embedding_policy = vector_embedding_policy
        self.indexing_policy = indexing_policy
        self.cosmos_container_properties = cosmos_container_properties
        self.cosmos_database_properties = cosmos_database_properties or {}
        self.full_text_search_enabled = full_text_search_enabled
        self.full_text_policy = full_text_policy
        self.connection_string = connection_string

        if full_text_search_enabled:
            if self.full_text_policy is None:
                self.full_text_policy = DEFAULT_FULL_TEXT_POLICY
            if not self.indexing_policy.get("fullTextIndexes"):
                self.indexing_policy = {
                    **self.indexing_policy,
                    "fullTextIndexes": [{"path": FULL_TEXT_PATH}],
                }

        self._partition_key_path = self._resolve_partition_key_path(cosmos_container_properties["partition_key"])
        self._client = cosmos_client
        self._container: ContainerProxy | None = None

    @staticmethod
    def _resolve_partition_key_path(partition_key: Any) -> str:
        if isinstance(partition_key, PartitionKey):
            path = partition_key.path
            return path[0] if isinstance(path, list) else path
        if isinstance(partition_key, str):
            return partition_key
        msg = "partition_key must be an azure.cosmos.PartitionKey or a path string such as '/id'."
        raise ValueError(msg)

    @staticmethod
    def _resolve_uri(uri: str | Secret) -> str:
        """Resolves an account endpoint provided either as a plain string or as a `Secret`."""
        if isinstance(uri, Secret):
            return str(uri.resolve_value())
        return uri

    @property
    def client(self) -> CosmosClient:
        if self._client is None:
            connection_string = self.connection_string.resolve_value() if self.connection_string else None
            if connection_string:
                self._client = CosmosClient.from_connection_string(connection_string, user_agent=USER_AGENT)
            else:
                endpoint = Secret.from_env_var(AZURE_COSMOS_NOSQL_ENDPOINT, strict=True).resolve_value()
                self._client = CosmosClient(str(endpoint), credential=DefaultAzureCredential(), user_agent=USER_AGENT)
        return self._client

    @property
    def container(self) -> ContainerProxy:
        if self._container is None:
            database = self.client.create_database_if_not_exists(
                id=self.database_name, **self.cosmos_database_properties
            )
            container_kwargs: dict[str, Any] = {}
            if self.full_text_search_enabled and self.full_text_policy is not None:
                container_kwargs["full_text_policy"] = self.full_text_policy
            self._container = database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path=self._partition_key_path),
                indexing_policy=self.indexing_policy,
                vector_embedding_policy=self.vector_embedding_policy,
                **container_kwargs,
            )
        return self._container

    @classmethod
    def from_connection_string(
        cls,
        connection_string: Secret = Secret.from_env_var(AZURE_COSMOS_NOSQL_CONNECTION_STRING),  # noqa: B008
        *,
        database_name: str,
        container_name: str,
        vector_embedding_policy: dict[str, Any],
        indexing_policy: dict[str, Any],
        cosmos_container_properties: dict[str, Any],
        cosmos_database_properties: dict[str, Any] | None = None,
        full_text_search_enabled: bool = False,
        full_text_policy: dict[str, Any] | None = None,
    ) -> AzureCosmosDBNoSqlDocumentStore:
        """Creates a store from a Cosmos DB connection string (serialization-friendly)."""
        return cls(
            connection_string=connection_string,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            full_text_search_enabled=full_text_search_enabled,
            full_text_policy=full_text_policy,
        )

    @classmethod
    def from_uri_and_key(
        cls,
        uri: str | Secret = Secret.from_env_var(AZURE_COSMOS_NOSQL_ENDPOINT),  # noqa: B008
        key: Secret = Secret.from_env_var(AZURE_COSMOS_NOSQL_KEY),  # noqa: B008
        *,
        database_name: str,
        container_name: str,
        vector_embedding_policy: dict[str, Any],
        indexing_policy: dict[str, Any],
        cosmos_container_properties: dict[str, Any],
        cosmos_database_properties: dict[str, Any] | None = None,
        full_text_search_enabled: bool = False,
        full_text_policy: dict[str, Any] | None = None,
    ) -> AzureCosmosDBNoSqlDocumentStore:
        """
        Creates a store from an account URI and key. The pair is stored as a connection string secret.

        :param uri: The account endpoint, as a plain string or a `Secret`. Defaults to the
            `AZURE_COSMOS_NOSQL_ENDPOINT` environment variable.
        :param key: The account key, wrapped in a `Secret`. Defaults to the `AZURE_COSMOS_NOSQL_KEY`
            environment variable.
        """
        connection_string = Secret.from_token(
            f"AccountEndpoint={cls._resolve_uri(uri)};AccountKey={key.resolve_value()};"
        )
        return cls(
            connection_string=connection_string,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            full_text_search_enabled=full_text_search_enabled,
            full_text_policy=full_text_policy,
        )

    @classmethod
    def from_aad_token(
        cls,
        uri: str | Secret = Secret.from_env_var(AZURE_COSMOS_NOSQL_ENDPOINT),  # noqa: B008
        *,
        credential: Any | None = None,
        database_name: str,
        container_name: str,
        vector_embedding_policy: dict[str, Any],
        indexing_policy: dict[str, Any],
        cosmos_container_properties: dict[str, Any],
        cosmos_database_properties: dict[str, Any] | None = None,
        full_text_search_enabled: bool = False,
        full_text_policy: dict[str, Any] | None = None,
    ) -> AzureCosmosDBNoSqlDocumentStore:
        """
        Creates a store authenticated with Microsoft Entra ID. Defaults to `DefaultAzureCredential`.

        :param uri: The account endpoint, as a plain string or a `Secret`. Defaults to the
            `AZURE_COSMOS_NOSQL_ENDPOINT` environment variable.
        :param credential: An Entra ID credential. Defaults to `DefaultAzureCredential`.
        """
        cosmos_client = CosmosClient(
            cls._resolve_uri(uri), credential=credential or DefaultAzureCredential(), user_agent=USER_AGENT
        )
        return cls(
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            full_text_search_enabled=full_text_search_enabled,
            full_text_policy=full_text_policy,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the document store to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            database_name=self.database_name,
            container_name=self.container_name,
            vector_embedding_policy=self.vector_embedding_policy,
            indexing_policy=self.indexing_policy,
            cosmos_container_properties={"partition_key": self._partition_key_path},
            cosmos_database_properties=self.cosmos_database_properties,
            full_text_search_enabled=self.full_text_search_enabled,
            full_text_policy=self.full_text_policy,
            connection_string=self.connection_string.to_dict() if self.connection_string else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AzureCosmosDBNoSqlDocumentStore:
        """
        Deserializes the document store from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized document store.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["connection_string"])
        container_properties = data["init_parameters"].get("cosmos_container_properties") or {}
        partition_key = container_properties.get("partition_key")
        if isinstance(partition_key, str):
            container_properties["partition_key"] = PartitionKey(path=partition_key)
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: The number of documents in the document store.
        """
        result = list(
            self.container.query_items(query="SELECT VALUE COUNT(1) FROM c", enable_cross_partition_query=True)
        )
        return cast(int, result[0]) if result else 0

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents into the Cosmos DB container.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises ValueError: If `documents` is not a list of `Document`.
        :raises DuplicateDocumentError: If a document with the same id already exists and `policy` is
            `DuplicatePolicy.FAIL`.
        :returns: The number of documents written.
        """
        if documents and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy in (DuplicatePolicy.NONE, DuplicatePolicy.FAIL):
            policy = DuplicatePolicy.FAIL

        written_docs = 0
        for doc in documents:
            item = self._haystack_doc_to_cosmos_doc(doc)
            if policy == DuplicatePolicy.OVERWRITE:
                self.container.upsert_item(item)
                written_docs += 1
            elif policy == DuplicatePolicy.SKIP:
                try:
                    self.container.create_item(item)
                    written_docs += 1
                except CosmosResourceExistsError:
                    logger.debug("Document %s already exists, skipping.", doc.id)
            else:  # DuplicatePolicy.FAIL
                try:
                    self.container.create_item(item)
                    written_docs += 1
                except CosmosResourceExistsError as exc:
                    msg = f"Document with id '{doc.id}' already exists in the document store."
                    raise DuplicateDocumentError(msg) from exc

        return written_docs

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents with the matching `document_ids` from the document store.

        The partition key value is assumed to be the document id (the recommended
        `partition_key=PartitionKey(path="/id")` configuration).

        :param document_ids: The document ids to delete.
        """
        if not document_ids:
            return
        for document_id in document_ids:
            self.container.delete_item(item=document_id, partition_key=document_id)

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the provided filters.

        For a detailed specification of the filters, refer to the
        [Haystack documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

        :param filters: The filters to apply.
        :returns: A list of Documents that match the given filters.
        """
        parameters: list[dict[str, Any]] = []
        query = "SELECT * FROM c"
        if filters:
            where_clause, parameters = _normalize_filters(filters)
            query += f" WHERE {where_clause}"

        items = list(self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True))
        return [self._cosmos_doc_to_haystack_doc(item) for item in items]

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Finds the documents most similar to `query_embedding` using vector similarity.

        :param query_embedding: Embedding of the query.
        :param filters: Optional metadata filters.
        :param top_k: How many documents to return.
        :returns: A list of Documents most similar to the given `query_embedding`, ordered by score.
        :raises ValueError: If `query_embedding` is empty.
        :raises DocumentStoreError: If the retrieval from Azure Cosmos DB for NoSQL fails.
        """
        if not query_embedding:
            msg = "query_embedding must not be empty"
            raise ValueError(msg)

        parameters: list[dict[str, Any]] = [
            {"name": "@top_k", "value": top_k},
            {"name": "@embedding", "value": query_embedding},
        ]

        query = (
            "SELECT TOP @top_k c.id, c.content, c.blob, c.meta, c.embedding, "
            "VectorDistance(c.embedding, @embedding) AS score FROM c"
        )
        if filters:
            where_clause, filter_parameters = _normalize_filters(filters)
            query += f" WHERE {where_clause}"
            parameters += filter_parameters
        query += " ORDER BY VectorDistance(c.embedding, @embedding)"

        try:
            items = list(
                self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)
            )
        except Exception as exc:
            msg = f"Retrieval of documents from Azure Cosmos DB for NoSQL failed: {exc}"
            raise DocumentStoreError(msg) from exc

        return [self._cosmos_doc_to_haystack_doc(item) for item in items]

    def _full_text_retrieval(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Finds the documents most relevant to `query_text` using BM25 full-text ranking.

        Uses `ORDER BY RANK FullTextScore(...)`, which merges and re-ranks results across all
        partitions. The returned documents are ordered by relevance; per-document BM25 scores are
        not exposed because Cosmos DB does not allow projecting `FullTextScore`.

        :param query_text: The free-text query. Tokenized on whitespace into search terms.
        :param filters: Optional metadata filters, applied as a `WHERE` clause before ranking.
        :param top_k: How many documents to return.
        :returns: A list of Documents ordered by full-text relevance.
        :raises ValueError: If `query_text` is empty.
        :raises DocumentStoreError: If full-text search is not enabled or the retrieval fails.
        """
        if not query_text or not query_text.strip():
            msg = "query_text must not be empty"
            raise ValueError(msg)
        self._ensure_full_text_enabled()

        parameters: list[dict[str, Any]] = [{"name": "@top_k", "value": top_k}]
        term_params = self._build_term_parameters(query_text, parameters)

        query = "SELECT TOP @top_k c.id, c.content, c.blob, c.meta, c.embedding FROM c"
        if filters:
            where_clause, filter_parameters = _normalize_filters(filters)
            query += f" WHERE {where_clause}"
            parameters += filter_parameters
        query += f" ORDER BY RANK FullTextScore(c.{FULL_TEXT_FIELD}, {term_params})"

        try:
            items = list(
                self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)
            )
        except Exception as exc:
            msg = f"Full-text retrieval from Azure Cosmos DB for NoSQL failed: {exc}"
            raise DocumentStoreError(msg) from exc

        return [self._cosmos_doc_to_haystack_doc(item) for item in items]

    def _hybrid_retrieval(
        self,
        query_embedding: list[float],
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        weights: list[float] | None = None,
    ) -> list[Document]:
        """
        Finds documents by fusing vector and full-text relevance with Reciprocal Rank Fusion (RRF).

        Uses `ORDER BY RANK RRF(FullTextScore(...), VectorDistance(...))`. The returned documents are
        ordered by the fused RRF rank; each document's `score` carries its raw `VectorDistance`
        similarity for reference.

        :param query_embedding: Embedding of the query.
        :param query_text: The free-text query. Tokenized on whitespace into search terms.
        :param filters: Optional metadata filters, applied as a `WHERE` clause before ranking.
        :param top_k: How many documents to return.
        :param weights: Optional per-component weights `[full_text_weight, vector_weight]`. When
            omitted, RRF weights all components equally.
        :returns: A list of Documents ordered by fused relevance.
        :raises ValueError: If `query_embedding` or `query_text` is empty.
        :raises DocumentStoreError: If full-text search is not enabled or the retrieval fails.
        """
        if not query_embedding:
            msg = "query_embedding must not be empty"
            raise ValueError(msg)
        if not query_text or not query_text.strip():
            msg = "query_text must not be empty"
            raise ValueError(msg)
        self._ensure_full_text_enabled()

        parameters: list[dict[str, Any]] = [
            {"name": "@top_k", "value": top_k},
            {"name": "@embedding", "value": query_embedding},
        ]
        term_params = self._build_term_parameters(query_text, parameters)

        query = (
            "SELECT TOP @top_k c.id, c.content, c.blob, c.meta, c.embedding, "
            "VectorDistance(c.embedding, @embedding) AS score FROM c"
        )
        if filters:
            where_clause, filter_parameters = _normalize_filters(filters)
            query += f" WHERE {where_clause}"
            parameters += filter_parameters

        full_text_score = f"FullTextScore(c.{FULL_TEXT_FIELD}, {term_params})"
        vector_distance = "VectorDistance(c.embedding, @embedding)"
        if weights:
            if len(weights) != _HYBRID_RRF_COMPONENTS:
                msg = "weights must contain exactly two values: [full_text_weight, vector_weight]."
                raise ValueError(msg)
            # Weights are coerced to floats and embedded as a literal array (Cosmos DB requires a
            # literal, not a parameter, for RRF weights). Numeric coercion keeps this injection-safe.
            literal_weights = ", ".join(str(float(w)) for w in weights)
            query += f" ORDER BY RANK RRF({full_text_score}, {vector_distance}, [{literal_weights}])"
        else:
            query += f" ORDER BY RANK RRF({full_text_score}, {vector_distance})"

        try:
            items = list(
                self.container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)
            )
        except Exception as exc:
            msg = f"Hybrid retrieval from Azure Cosmos DB for NoSQL failed: {exc}"
            raise DocumentStoreError(msg) from exc

        return [self._cosmos_doc_to_haystack_doc(item) for item in items]

    def _ensure_full_text_enabled(self) -> None:
        if not self.full_text_search_enabled:
            msg = (
                "Full-text search is not enabled on this document store. "
                "Create the store with full_text_search_enabled=True to use full-text or hybrid retrieval."
            )
            raise DocumentStoreError(msg)

    @staticmethod
    def _build_term_parameters(query_text: str, parameters: list[dict[str, Any]]) -> str:
        """Appends one parameter per whitespace-delimited term and returns the comma-joined names."""
        names = []
        for i, term in enumerate(query_text.split()):
            name = f"@ft_term_{i}"
            parameters.append({"name": name, "value": term})
            names.append(name)
        return ", ".join(names)

    @staticmethod
    def _haystack_doc_to_cosmos_doc(document: Document) -> dict[str, Any]:
        """Converts a Haystack Document into a Cosmos DB item, dropping unsupported fields."""
        item = document.to_dict(flatten=False)
        if item.get("sparse_embedding"):
            logger.warning(
                "Document %s has the `sparse_embedding` field set, but storing sparse embeddings in "
                "Azure Cosmos DB for NoSQL is not currently supported. The `sparse_embedding` field will be ignored.",
                document.id,
            )
        item.pop("sparse_embedding", None)
        return item

    @staticmethod
    def _cosmos_doc_to_haystack_doc(cosmos_doc: dict[str, Any]) -> Document:
        """
        Converts an item coming out of Cosmos DB into a Haystack Document.

        :param cosmos_doc: A dictionary representing a document as stored in Cosmos DB.
        :returns: A Haystack Document.
        """
        # Remove Cosmos DB system-generated fields that don't belong to a Haystack Document.
        for system_field in ("_rid", "_self", "_etag", "_attachments", "_ts"):
            cosmos_doc.pop(system_field, None)
        return Document.from_dict(cosmos_doc)
