# SPDX-FileCopyrightText: 2024-present Microsoft Corporation
#
# SPDX-License-Identifier: MIT
import pytest
from haystack.errors import FilterError

from haystack_azure_cosmosdb.filters import _normalize_filters


class TestNormalizeFilters:
    def test_raises_on_non_dict(self):
        with pytest.raises(FilterError):
            _normalize_filters("not a dict")

    def test_raises_on_invalid_syntax(self):
        with pytest.raises(FilterError):
            _normalize_filters({"value": 5})

    def test_equal(self):
        clause, params = _normalize_filters({"field": "meta.number", "operator": "==", "value": 100})
        assert clause == "c.meta.number = @param0"
        assert params == [{"name": "@param0", "value": 100}]

    def test_equal_none_uses_is_null(self):
        clause, params = _normalize_filters({"field": "meta.number", "operator": "==", "value": None})
        assert clause == "IS_NULL(c.meta.number)"
        assert params == []

    def test_not_equal(self):
        clause, params = _normalize_filters({"field": "meta.chapter", "operator": "!=", "value": "intro"})
        assert clause == "(NOT IS_DEFINED(c.meta.chapter) OR c.meta.chapter != @param0)"
        assert params == [{"name": "@param0", "value": "intro"}]

    def test_comparison_operators(self):
        for operator, symbol in [(">", ">"), (">=", ">="), ("<", "<"), ("<=", "<=")]:
            clause, params = _normalize_filters({"field": "meta.number", "operator": operator, "value": 5})
            assert clause == f"c.meta.number {symbol} @param0"
            assert params == [{"name": "@param0", "value": 5}]

    def test_in(self):
        clause, params = _normalize_filters({"field": "meta.name", "operator": "in", "value": ["a", "b"]})
        assert clause == "c.meta.name IN (@param0, @param1)"
        assert params == [{"name": "@param0", "value": "a"}, {"name": "@param1", "value": "b"}]

    def test_not_in(self):
        clause, params = _normalize_filters({"field": "meta.name", "operator": "not in", "value": ["a", "b"]})
        assert clause == "c.meta.name NOT IN (@param0, @param1)"
        assert params == [{"name": "@param0", "value": "a"}, {"name": "@param1", "value": "b"}]

    def test_in_requires_list(self):
        with pytest.raises(FilterError):
            _normalize_filters({"field": "meta.name", "operator": "in", "value": "a"})

    def test_comparison_rejects_list_value(self):
        with pytest.raises(FilterError):
            _normalize_filters({"field": "meta.number", "operator": ">", "value": [1, 2]})

    def test_comparison_rejects_non_iso_string(self):
        with pytest.raises(FilterError):
            _normalize_filters({"field": "meta.name", "operator": ">", "value": "not-a-date"})

    def test_and(self):
        clause, params = _normalize_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.chapter", "operator": "==", "value": "intro"},
                ],
            }
        )
        assert clause == "(c.meta.number = @param0 AND c.meta.chapter = @param1)"
        assert params == [{"name": "@param0", "value": 100}, {"name": "@param1", "value": "intro"}]

    def test_or(self):
        clause, _ = _normalize_filters(
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.number", "operator": "==", "value": 90},
                ],
            }
        )
        assert clause == "(c.meta.number = @param0 OR c.meta.number = @param1)"

    def test_not(self):
        clause, _ = _normalize_filters(
            {"operator": "NOT", "conditions": [{"field": "meta.number", "operator": "==", "value": 100}]}
        )
        assert clause == "NOT (c.meta.number = @param0)"

    def test_nested_logical(self):
        clause, params = _normalize_filters(
            {
                "operator": "OR",
                "conditions": [
                    {
                        "operator": "AND",
                        "conditions": [
                            {"field": "meta.number", "operator": "==", "value": 100},
                            {"field": "meta.chapter", "operator": "==", "value": "intro"},
                        ],
                    },
                    {"field": "meta.page", "operator": "==", "value": "90"},
                ],
            }
        )
        assert clause == "((c.meta.number = @param0 AND c.meta.chapter = @param1) OR c.meta.page = @param2)"
        assert [p["value"] for p in params] == [100, "intro", "90"]

    def test_unknown_logical_operator(self):
        with pytest.raises(FilterError):
            _normalize_filters({"operator": "XOR", "conditions": []})

    def test_unknown_comparison_operator(self):
        with pytest.raises(FilterError):
            _normalize_filters({"field": "meta.number", "operator": "~=", "value": 1})
