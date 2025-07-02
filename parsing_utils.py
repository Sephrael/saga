# parsing_utils.py
"""Backwards compatibility wrapper for parsing utilities."""

from parsing import (
    ParseError,
    _get_entity_type_and_name_from_text,
    parse_rdf_triples_with_rdflib,
)

__all__ = [
    "ParseError",
    "_get_entity_type_and_name_from_text",
    "parse_rdf_triples_with_rdflib",
]
