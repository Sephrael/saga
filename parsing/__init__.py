# parsing/__init__.py
"""Common parsing utilities for SAGA."""

from __future__ import annotations

import json
import logging
from typing import Any

import kg_constants as kg_keys
import structlog
import utils
from kg_maintainer.models import CharacterProfile, WorldItem

from models import ProblemDetail

logger = structlog.get_logger(__name__)


class ParseError(Exception):
    """Custom exception for parsing errors."""


def _get_entity_type_and_name_from_text(entity_text: str) -> dict[str, str | None]:
    """Parse ``EntityType:EntityName`` or just ``EntityName`` strings."""
    name_part = entity_text
    type_part: str | None = None
    if ":" in entity_text:
        parts = entity_text.split(":", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            type_part = parts[0].strip()
            name_part = parts[1].strip()
        elif parts[0].strip():
            type_part = parts[0].strip()
            name_part = parts[1].strip() if len(parts) > 1 else ""
    return {
        "type": type_part if type_part else None,
        "name": name_part.strip() if name_part else None,
    }


def _parse_relationships(rels_val: Any) -> dict[str, str]:
    """Return normalized relationship dictionary."""
    rels_dict: dict[str, str] = {}
    if isinstance(rels_val, list):
        for rel_entry in rels_val:
            if isinstance(rel_entry, str):
                if ":" in rel_entry:
                    parts = rel_entry.split(":", 1)
                    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                        rels_dict[parts[0].strip()] = parts[1].strip()
                    elif parts[0].strip():
                        rels_dict[parts[0].strip()] = "related"
                elif rel_entry.strip():
                    rels_dict[rel_entry.strip()] = "related"
            elif isinstance(rel_entry, dict):
                target_name = rel_entry.get("name")
                detail = rel_entry.get("detail", "related")
                if target_name and isinstance(target_name, str) and target_name.strip():
                    rels_dict[target_name] = detail
    elif isinstance(rels_val, dict):
        rels_dict = {str(k): str(v) for k, v in rels_val.items()}
    return rels_dict


def _ensure_dev_key(char_name: str, attrs: dict[str, Any], chapter_number: int) -> None:
    """Standardize development event key in ``attrs``."""
    dev_key_standard = kg_keys.development_key(chapter_number)
    specific_dev_key = next(
        (k for k in attrs if k.lower().replace(" ", "_") == dev_key_standard),
        None,
    )
    if specific_dev_key and specific_dev_key != dev_key_standard:
        attrs[dev_key_standard] = attrs.pop(specific_dev_key)

    has_other = any(
        k not in {"modification_proposal", dev_key_standard} and v
        for k, v in attrs.items()
    )
    if not attrs.get(dev_key_standard) and has_other:
        attrs[dev_key_standard] = (
            f"Character '{char_name}' details updated in Chapter {chapter_number}."
        )


def _parse_triple_line(
    line: str, line_num: int, logger_obj: logging.Logger
) -> tuple[dict[str, str | None], str, str] | None:
    """Return parsed parts if line is valid; otherwise ``None``."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("//"):
        return None
    parts = [p.strip() for p in stripped.split("|")]
    if len(parts) < 3:
        logger_obj.warning(
            "Line %s: Malformed triple (expected 3 parts): '%s'",
            line_num + 1,
            line,
        )
        return None
    subject_text, predicate_text, object_text = parts[:3]
    if len(parts) > 3:
        logger_obj.debug(
            "Line %s: extra parts ignored: '%s' from '%s'",
            line_num + 1,
            " | ".join(parts[3:]),
            line,
        )
    subject_details = _get_entity_type_and_name_from_text(subject_text)
    predicate_str = predicate_text.strip().upper().replace(" ", "_")
    if not subject_details.get("name") or not predicate_str:
        logger_obj.warning(
            "Line %s: Missing subject name or predicate: S='%s', P='%s'",
            line_num + 1,
            subject_text,
            predicate_text,
        )
        return None
    return subject_details, predicate_str, object_text


def _parse_object_payload(
    object_text: str, line_num: int, logger_obj: logging.Logger
) -> tuple[dict[str, str | None] | None, str | None, bool]:
    """Return object payload components for a triple."""
    object_entity_payload: dict[str, str | None] | None = None
    object_literal_payload: str | None = None
    is_literal_object = True
    if ":" in object_text:
        obj_parts_check = object_text.split(":", 1)
        if (
            len(obj_parts_check) == 2
            and obj_parts_check[0].strip()
            and obj_parts_check[1].strip()
        ):
            potential_obj_type = obj_parts_check[0].strip()
            if potential_obj_type[0].isupper() and " " not in potential_obj_type:
                object_entity_payload = _get_entity_type_and_name_from_text(object_text)
                is_literal_object = False
    if is_literal_object:
        object_literal_payload = object_text.strip().strip('"').strip("'")
    if not is_literal_object and (
        not object_entity_payload or not object_entity_payload.get("name")
    ):
        logger_obj.debug(
            "Line %s: Object '%s' looked like entity but parsed no name. Reverting to literal.",
            line_num + 1,
            object_text,
        )
        object_literal_payload = object_text.strip()
        is_literal_object = True
        object_entity_payload = None
    return object_entity_payload, object_literal_payload, is_literal_object


def parse_rdf_triples_with_rdflib(
    text_block: str,
    rdf_format: str = "turtle",
    base_uri: str = "http://example.org/saga/",
) -> list[dict[str, Any]]:
    """Parse plain text triples into structured dictionaries."""
    logger_func = logging.getLogger(__name__)
    triples_list: list[dict[str, Any]] = []
    if not text_block or not text_block.strip():
        return triples_list

    lines = text_block.strip().splitlines()
    for line_num, line in enumerate(lines):
        parsed_line = _parse_triple_line(line, line_num, logger_func)
        if not parsed_line:
            continue
        subject_details, predicate_str, object_text = parsed_line
        (
            object_entity_payload,
            object_literal_payload,
            is_literal_object,
        ) = _parse_object_payload(object_text, line_num, logger_func)
        triples_list.append(
            {
                "subject": subject_details,
                "predicate": predicate_str,
                "object_entity": object_entity_payload,
                "object_literal": object_literal_payload,
                "is_literal_object": is_literal_object,
            }
        )
    return triples_list


def _normalize_attributes(
    attributes_dict: dict[str, Any],
    key_map: dict[str, str],
    list_keys: list[str],
) -> dict[str, Any]:
    normalized_attrs: dict[str, Any] = {}
    if not isinstance(attributes_dict, dict):
        logger.warning(
            "Input to _normalize_attributes was not a dict: %s", type(attributes_dict)
        )
        return {}

    for key, value in attributes_dict.items():
        normalized_llm_key = key.lower().replace(" ", "_")
        mapped_key = key_map.get(normalized_llm_key, normalized_llm_key)
        if mapped_key in list_keys:
            if isinstance(value, list | dict):
                normalized_attrs[mapped_key] = value
            elif isinstance(value, str):
                normalized_attrs[mapped_key] = [
                    v.strip() for v in value.split(",") if v.strip()
                ]
            elif value is None:
                normalized_attrs[mapped_key] = []
            else:
                normalized_attrs[mapped_key] = [value]
        else:
            normalized_attrs[mapped_key] = value

    for l_key in list_keys:
        if l_key not in normalized_attrs:
            normalized_attrs[l_key] = []
        elif not isinstance(normalized_attrs[l_key], list):
            if isinstance(normalized_attrs[l_key], dict):
                continue
            if (
                normalized_attrs[l_key] is not None
                and str(normalized_attrs[l_key]).strip()
            ):
                normalized_attrs[l_key] = [str(normalized_attrs[l_key])]
            else:
                normalized_attrs[l_key] = []

    return normalized_attrs


CHAR_UPDATE_KEY_MAP = {
    "desc": "description",
    "description": "description",
    "traits": "traits",
    "relationships": "relationships",
    "status": "status",
    "modification proposal": "modification_proposal",
}

CHAR_UPDATE_LIST_INTERNAL_KEYS = ["traits", "relationships", "aliases"]

WORLD_UPDATE_DETAIL_KEY_MAP = {
    "desc": "description",
    "description": "description",
    "atmosphere": "atmosphere",
    "goals": "goals",
    "rules": "rules",
    "key elements": "key_elements",
    "traits": "traits",
    "modification proposal": "modification_proposal",
}

WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS = ["goals", "rules", "key_elements", "traits"]


def _process_single_character_attributes(
    char_name: str,
    char_attributes_llm: dict[str, Any],
    chapter_number: int,
) -> CharacterProfile | None:
    """Normalizes and creates a CharacterProfile from raw LLM attributes."""
    if not char_name or not isinstance(char_attributes_llm, dict):
        logger.warning(
            "Skipping character with invalid name or attributes: Name='%s', Attrs_Type='%s'",
            char_name,
            type(char_attributes_llm),
        )
        return None

    processed_attributes = _normalize_attributes(
        char_attributes_llm,
        CHAR_UPDATE_KEY_MAP,
        CHAR_UPDATE_LIST_INTERNAL_KEYS,
    )

    traits_val = processed_attributes.get("traits", [])
    if isinstance(traits_val, list):
        processed_attributes["traits"] = [
            utils.normalize_trait_name(t)
            for t in traits_val
            if isinstance(t, str) and utils.normalize_trait_name(t)
        ]

    processed_attributes["relationships"] = _parse_relationships(
        processed_attributes.get("relationships")
    )

    _ensure_dev_key(char_name, processed_attributes, chapter_number)

    try:
        return CharacterProfile.from_dict(char_name, processed_attributes)
    except Exception as e:
        logger.error(
            "Error creating CharacterProfile for '%s': %s. Attributes: %s",
            char_name,
            e,
            processed_attributes,
            exc_info=True,
        )
        return None


def parse_unified_character_updates(
    json_text_block: str, chapter_number: int
) -> dict[str, CharacterProfile]:
    """Parse character update JSON provided by the LLM."""
    char_updates: dict[str, CharacterProfile] = {}
    if not json_text_block.strip():
        return char_updates

    try:
        parsed_data = json.loads(json_text_block)
        if not isinstance(parsed_data, dict):
            logger.error(
                "Character updates JSON was not a dictionary. Received: %s",
                type(parsed_data),
            )
            return char_updates
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse character updates JSON: %s. Input: %s...",
            e,
            json_text_block[:500],
        )
        return char_updates

    for char_name, char_attributes_llm in parsed_data.items():
        profile = _process_single_character_attributes(
            char_name, char_attributes_llm, chapter_number
        )
        if profile:
            char_updates[char_name] = profile
    return char_updates


def _process_world_item_attributes(
    category_name: str,
    item_name: str,
    item_attributes_llm: dict[str, Any],
    chapter_number: int,
    is_overview: bool = False,
) -> WorldItem | None:
    """Normalizes attributes and creates a WorldItem instance."""
    processed_item_details = _normalize_attributes(
        item_attributes_llm,
        WORLD_UPDATE_DETAIL_KEY_MAP,
        WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
    )
    elaboration_key_standard = kg_keys.elaboration_key(chapter_number)

    has_meaningful_attrs = any(
        k not in ["modification_proposal", elaboration_key_standard] and v
        for k, v in processed_item_details.items()
    )
    if is_overview:  # Overview specific elaboration
        if any(
            k != "modification_proposal" for k in processed_item_details
        ):  # Check if any non-proposal key exists
            if not processed_item_details.get(elaboration_key_standard):
                processed_item_details[elaboration_key_standard] = (
                    f"Overall world overview updated in Chapter {chapter_number}."
                )
    elif has_meaningful_attrs and not processed_item_details.get(
        elaboration_key_standard
    ):
        processed_item_details[elaboration_key_standard] = (
            f"Item '{item_name}' in category '{category_name}' updated in Chapter {chapter_number}."
        )

    try:
        return WorldItem.from_dict(
            category_name,
            item_name,
            processed_item_details,
        )
    except Exception as e:
        logger.error(
            "Error creating WorldItem for '%s' in category '%s': %s. Details: %s",
            item_name,
            category_name,
            e,
            processed_item_details,
            exc_info=True,
        )
        return None


def _parse_world_category_items(
    category_name_llm: str,
    items_llm: dict[str, Any],
    chapter_number: int,
    results: dict[str, dict[str, WorldItem]],
) -> None:
    """Parses items within a single world category."""
    category_dict_by_item_name: dict[str, WorldItem] = {}

    if category_name_llm.lower() in {"overview", "_overview_"}:
        overview_item = _process_world_item_attributes(
            category_name_llm, "_overview_", items_llm, chapter_number, is_overview=True
        )
        if overview_item:
            results.setdefault(category_name_llm, {})["_overview_"] = overview_item
    else:
        # Handle cases where items_llm might be a flat dict for a single item
        # or a dict of dicts for multiple items.
        # Heuristic: if all values in items_llm are not dicts, assume it's a single item's attributes.
        if items_llm and all(not isinstance(v, dict) for v in items_llm.values()):
            # This structure implies items_llm is actually the attribute dict for an item whose name is category_name_llm
            # This is a fallback for potentially malformed LLM output.
            # Consider if this case is truly expected or if LLM output should be stricter.
            world_item_instance = _process_world_item_attributes(
                category_name_llm,  # The category is used as item name here
                category_name_llm,  # And also as category name
                items_llm,
                chapter_number,
            )
            if world_item_instance:
                category_dict_by_item_name[world_item_instance.name] = (
                    world_item_instance
                )
        else:  # Standard case: items_llm is a dict of item_name: attributes_dict
            for item_name_llm, item_attributes_llm in items_llm.items():
                if not item_name_llm or not isinstance(item_attributes_llm, dict):
                    logger.warning(
                        "Skipping item with invalid name or attributes in category '%s': Name='%s', AttrType='%s'",
                        category_name_llm,
                        item_name_llm,
                        type(item_attributes_llm),
                    )
                    continue
                world_item_instance = _process_world_item_attributes(
                    category_name_llm,
                    item_name_llm,
                    item_attributes_llm,
                    chapter_number,
                )
                if world_item_instance:
                    category_dict_by_item_name[world_item_instance.name] = (
                        world_item_instance
                    )
        if category_dict_by_item_name:
            results[category_name_llm] = category_dict_by_item_name


def parse_unified_world_updates(
    json_text_block: str, chapter_number: int
) -> dict[str, dict[str, WorldItem]]:
    """Parse world update JSON provided by the LLM."""
    if not json_text_block.strip():
        return {}

    try:
        parsed_data = json.loads(json_text_block)
        if not isinstance(parsed_data, dict):
            logger.error(
                "World updates JSON was not a dictionary. Received: %s",
                type(parsed_data),
            )
            return {}
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse world updates JSON: %s. Input: %s...",
            e,
            json_text_block[:500],
        )
        return {}

    results: dict[str, dict[str, WorldItem]] = {}
    for category_name_llm, items_llm in parsed_data.items():
        if not isinstance(
            items_llm, dict
        ):  # items_llm should be a dict (either of item attrs or item names to item attrs)
            logger.warning(
                "Skipping category '%s' as its content is not a dictionary. Type: %s",
                category_name_llm,
                type(items_llm),
            )
            continue
        _parse_world_category_items(
            category_name_llm, items_llm, chapter_number, results
        )
    return results


def _load_and_validate_problem_json(
    text: str, category: str | None = None
) -> list[Any] | ProblemDetail:
    """Loads JSON text and validates if it's a list of problems.
    Returns the list of raw problem dicts if valid, or a ProblemDetail error object.
    """
    if not text or not text.strip():
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            # Handle cases where LLM might return a status object instead of a list
            if "status" in data and "no significant" in str(data["status"]).lower():
                return []  # No problems found
            if "problems" in data and isinstance(data["problems"], list):
                data = data["problems"]  # Extract the list
        if not isinstance(data, list):
            logger.error(
                "LLM output was not a list of problems, but type: %s", type(data)
            )
            return ProblemDetail(
                issue_category=category or "meta",
                problem_description="LLM output was not a list of problems as expected.",
                quote_from_original_text="N/A - Malformed LLM output structure",
                suggested_fix_focus="Ensure LLM outputs a JSON list of problem objects.",
            )
        return data
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to decode problem list JSON: %s. Input: %s...", exc, text[:200]
        )
        return ProblemDetail(
            issue_category=category or "meta",
            problem_description=f"Invalid JSON from LLM: {exc}",
            quote_from_original_text="N/A - Invalid JSON",
            suggested_fix_focus="Ensure LLM outputs valid JSON.",
        )


def _create_problem_detail_from_item(
    item: Any, default_category: str | None = None
) -> ProblemDetail | None:
    """Creates a ProblemDetail object from a parsed item, with validation."""
    if not isinstance(item, dict):
        logger.warning("Problem item is not a dict: %s", item)
        return None

    # Use .get() with defaults for all fields to prevent KeyErrors
    return ProblemDetail(
        issue_category=item.get("issue_category", default_category or "meta"),
        problem_description=item.get(
            "problem_description", "N/A - Missing description"
        ),
        quote_from_original_text=item.get(
            "quote_from_original_text", "N/A - General Issue"
        ),
        quote_char_start=item.get(
            "quote_char_start"
        ),  # Defaults to None if not present
        quote_char_end=item.get("quote_char_end"),
        sentence_char_start=item.get("sentence_char_start"),
        sentence_char_end=item.get("sentence_char_end"),
        suggested_fix_focus=item.get("suggested_fix_focus", "N/A - Missing suggestion"),
        rewrite_instruction=item.get("rewrite_instruction"),
        severity=item.get("severity"),
        related_spans=item.get("related_spans"),
    )


def parse_problem_list(text: str, category: str | None = None) -> list[ProblemDetail]:
    """Parse a JSON list of problem details."""
    loaded_data = _load_and_validate_problem_json(text, category)

    if isinstance(loaded_data, ProblemDetail):  # Error case from validation
        return [loaded_data]
    if not loaded_data:  # Empty list
        return []

    problems: list[ProblemDetail] = []
    for item in loaded_data:
        problem_detail = _create_problem_detail_from_item(item, category)
        if problem_detail:
            problems.append(problem_detail)
    return problems
