# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import datetime
from typing import Any

from haystack.errors import FilterError

UNSUPPORTED_TYPES_FOR_COMPARISON = (list,)

# Parameters returned alongside a WHERE clause. Each entry follows the Azure
# Cosmos DB parameterized-query format: ``{"name": "@paramN", "value": <value>}``.
Parameters = list[dict[str, Any]]


def _normalize_filters(filters: dict[str, Any]) -> tuple[str, Parameters]:
    """
    Converts Haystack filters into an Azure Cosmos DB for NoSQL ``WHERE`` clause.

    The returned clause references the document alias ``c`` and uses parameterized
    values (``@param0``, ``@param1``, ...) to avoid SQL injection. The parameters are
    returned as a list ready to be passed to ``container.query_items(..., parameters=...)``.

    :param filters: The Haystack filters to convert.
    :returns: A tuple of the ``WHERE`` clause body (without the ``WHERE`` keyword) and
        the list of query parameters.
    :raises FilterError: If the filter syntax is invalid.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" not in filters and "operator" not in filters and "conditions" not in filters:
        msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
        raise FilterError(msg)

    parameters: Parameters = []
    clause = _parse_condition(filters, parameters)
    return clause, parameters


def _parse_condition(condition: dict[str, Any], parameters: Parameters) -> str:
    if "field" in condition:
        return _parse_comparison_condition(condition, parameters)
    return _parse_logical_condition(condition, parameters)


def _parse_logical_condition(condition: dict[str, Any], parameters: Parameters) -> str:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    parsed = [_parse_condition(c, parameters) for c in condition["conditions"]]

    if operator == "AND":
        return "(" + " AND ".join(parsed) + ")"
    if operator == "OR":
        return "(" + " OR ".join(parsed) + ")"
    if operator == "NOT":
        return "NOT (" + " AND ".join(parsed) + ")"

    msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR', 'NOT'"
    raise FilterError(msg)


def _parse_comparison_condition(condition: dict[str, Any], parameters: Parameters) -> str:
    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    value: Any = condition["value"]

    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise FilterError(msg)

    return COMPARISON_OPERATORS[operator](_column(field), value, parameters)


def _column(field: str) -> str:
    """Maps a Haystack field name (e.g. ``meta.year``) to a Cosmos DB column (``c.meta.year``)."""
    return f"c.{field}"


def _add_parameter(value: Any, parameters: Parameters) -> str:
    name = f"@param{len(parameters)}"
    parameters.append({"name": name, "value": value})
    return name


def _validate_type_for_comparison(value: Any) -> None:
    msg = f"Can't compare {type(value)} using operators '>', '>=', '<', '<='."
    if isinstance(value, UNSUPPORTED_TYPES_FOR_COMPARISON):
        raise FilterError(msg)
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg += "\nStrings are only comparable if they are ISO formatted dates."
            raise FilterError(msg) from exc


def _equal(column: str, value: Any, parameters: Parameters) -> str:
    if value is None:
        return f"IS_NULL({column})"
    return f"{column} = {_add_parameter(value, parameters)}"


def _not_equal(column: str, value: Any, parameters: Parameters) -> str:
    if value is None:
        return f"NOT IS_NULL({column})"
    # ``!=`` should also match documents where the field is absent, mirroring Haystack semantics.
    return f"(NOT IS_DEFINED({column}) OR {column} != {_add_parameter(value, parameters)})"


def _greater_than(column: str, value: Any, parameters: Parameters) -> str:
    _validate_type_for_comparison(value)
    return f"{column} > {_add_parameter(value, parameters)}"


def _greater_than_equal(column: str, value: Any, parameters: Parameters) -> str:
    _validate_type_for_comparison(value)
    return f"{column} >= {_add_parameter(value, parameters)}"


def _less_than(column: str, value: Any, parameters: Parameters) -> str:
    _validate_type_for_comparison(value)
    return f"{column} < {_add_parameter(value, parameters)}"


def _less_than_equal(column: str, value: Any, parameters: Parameters) -> str:
    _validate_type_for_comparison(value)
    return f"{column} <= {_add_parameter(value, parameters)}"


def _in(column: str, value: Any, parameters: Parameters) -> str:
    if not isinstance(value, list):
        msg = f"{column}'s value must be a list when using 'in' comparator in Azure Cosmos DB for NoSQL"
        raise FilterError(msg)
    names = ", ".join(_add_parameter(v, parameters) for v in value)
    return f"{column} IN ({names})"


def _not_in(column: str, value: Any, parameters: Parameters) -> str:
    if not isinstance(value, list):
        msg = f"{column}'s value must be a list when using 'not in' comparator in Azure Cosmos DB for NoSQL"
        raise FilterError(msg)
    names = ", ".join(_add_parameter(v, parameters) for v in value)
    return f"{column} NOT IN ({names})"


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
}
