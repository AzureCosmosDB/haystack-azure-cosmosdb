# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import random
from typing import Any

from haystack.dataclasses import Document


def get_vector_indexing_policy(embedding_type: str = "quantizedFlat") -> dict[str, Any]:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": embedding_type}],
    }


def get_vector_embedding_policy(
    data_type: str = "float32", dimensions: int = 768, distance_function: str = "cosine"
) -> dict[str, Any]:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": data_type,
                "dimensions": dimensions,
                "distanceFunction": distance_function,
            }
        ]
    }


def get_documents() -> list[Document]:
    return [
        Document(
            content="Document A",
            meta={"name": "name_A", "page": "100", "chapter": "intro", "number": 2, "date": "1969-07-21T20:17:40"},
            embedding=_random_embeddings(768),
        ),
        Document(
            content="Document B",
            meta={"name": "name_B", "page": "123", "chapter": "abstract", "number": -2, "date": "1972-12-11T19:54:58"},
            embedding=_random_embeddings(768),
        ),
        Document(
            content="Document C",
            meta={
                "name": "name_C",
                "page": "90",
                "chapter": "conclusion",
                "number": -10,
                "date": "1989-11-09T17:53:00",
            },
            embedding=_random_embeddings(768),
        ),
    ]


def _random_embeddings(n: int) -> list[float]:
    return [random.random() for _ in range(n)]
