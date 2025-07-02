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
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            logger_func.warning(
                "Line %s: Malformed triple (expected 3 parts): '%s'",
                line_num + 1,
                line,
            )
            continue

        subject_text, predicate_text, object_text = parts[:3]
        if len(parts) > 3:
            logger_func.debug(
                "Line %s: extra parts ignored: '%s' from '%s'",
                line_num + 1,
                " | ".join(parts[3:]),
                line,
            )

        subject_details = _get_entity_type_and_name_from_text(subject_text)
        predicate_str = predicate_text.strip().upper().replace(" ", "_")
        if not subject_details.get("name") or not predicate_str:
            logger_func.warning(
                "Line %s: Missing subject name or predicate: S='%s', P='%s'",
                line_num + 1,
                subject_text,
                predicate_text,
            )
            continue

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
                    object_entity_payload = _get_entity_type_and_name_from_text(
                        object_text
                    )
                    is_literal_object = False
        if is_literal_object:
            object_literal_payload = object_text.strip()
            if object_literal_payload.startswith(
                '"'
            ) and object_literal_payload.endswith('"'):
                object_literal_payload = object_literal_payload[1:-1]
            if object_literal_payload.startswith(
                "'"
            ) and object_literal_payload.endswith("'"):
                object_literal_payload = object_literal_payload[1:-1]
        if not is_literal_object and (
            not object_entity_payload or not object_entity_payload.get("name")
        ):
            logger_func.debug(
                "Line %s: Object '%s' looked like entity but parsed no name. Reverting to literal.",
                line_num + 1,
                object_text,
            )
            object_literal_payload = object_text.strip()
            is_literal_object = True
            object_entity_payload = None

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
        if not char_name or not isinstance(char_attributes_llm, dict):
            logger.warning(
                "Skipping character with invalid name or attributes: Name='%s', Attrs_Type='%s'",
                char_name,
                type(char_attributes_llm),
            )
            continue

        processed_char_attributes = _normalize_attributes(
            char_attributes_llm,
            CHAR_UPDATE_KEY_MAP,
            CHAR_UPDATE_LIST_INTERNAL_KEYS,
        )

        traits_val = processed_char_attributes.get("traits", [])
        if isinstance(traits_val, list):
            processed_char_attributes["traits"] = [
                utils.normalize_trait_name(t)
                for t in traits_val
                if isinstance(t, str) and utils.normalize_trait_name(t)
            ]

        rels_val = processed_char_attributes.get("relationships")
        if isinstance(rels_val, list):
            rels_list = rels_val
            rels_dict: dict[str, str] = {}
            for rel_entry in rels_list:
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
                    if (
                        target_name
                        and isinstance(target_name, str)
                        and target_name.strip()
                    ):
                        rels_dict[target_name] = detail
            processed_char_attributes["relationships"] = rels_dict
        elif isinstance(rels_val, dict):
            processed_char_attributes["relationships"] = {
                str(k): str(v) for k, v in rels_val.items()
            }
        else:
            processed_char_attributes["relationships"] = {}

        dev_key_standard = kg_keys.development_key(chapter_number)
        specific_dev_key_from_llm = next(
            (
                k
                for k in processed_char_attributes
                if k.lower().replace(" ", "_") == dev_key_standard
            ),
            None,
        )
        if specific_dev_key_from_llm and specific_dev_key_from_llm != dev_key_standard:
            processed_char_attributes[dev_key_standard] = processed_char_attributes.pop(
                specific_dev_key_from_llm
            )

        has_other_meaningful_attrs = any(
            k not in ["modification_proposal", dev_key_standard] and v
            for k, v in processed_char_attributes.items()
        )
        if (
            not processed_char_attributes.get(dev_key_standard)
            and has_other_meaningful_attrs
        ):
            processed_char_attributes[dev_key_standard] = (
                f"Character '{char_name}' details updated in Chapter {chapter_number}."
            )

        try:
            char_updates[char_name] = CharacterProfile.from_dict(
                char_name, processed_char_attributes
            )
        except Exception as e:
            logger.error(
                "Error creating CharacterProfile for '%s': %s. Attributes: %s",
                char_name,
                e,
                processed_char_attributes,
                exc_info=True,
            )
    return char_updates


def parse_unified_world_updates(
    json_text_block: str, chapter_number: int
) -> dict[str, dict[str, WorldItem]]:
    """Parse world update JSON provided by the LLM."""
    world_updates: dict[str, dict[str, WorldItem]] = {}
    if not json_text_block.strip():
        return world_updates
    try:
        parsed_data = json.loads(json_text_block)
        if not isinstance(parsed_data, dict):
            logger.error(
                "World updates JSON was not a dictionary. Received: %s",
                type(parsed_data),
            )
            return world_updates
    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse world updates JSON: %s. Input: %s...",
            e,
            json_text_block[:500],
        )
        return world_updates

    results: dict[str, dict[str, WorldItem]] = {}
    for category_name_llm, items_llm in parsed_data.items():
        if not isinstance(items_llm, dict):
            logger.warning(
                "Skipping category '%s' as its content is not a dictionary of items.",
                category_name_llm,
            )
            continue

        category_dict_by_item_name: dict[str, WorldItem] = {}
        elaboration_key_standard = kg_keys.elaboration_key(chapter_number)

        if category_name_llm.lower() in {"overview", "_overview_"}:
            processed_overview_details = _normalize_attributes(
                items_llm,
                WORLD_UPDATE_DETAIL_KEY_MAP,
                WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
            )
            if any(k != "modification_proposal" for k in processed_overview_details):
                if not processed_overview_details.get(elaboration_key_standard):
                    processed_overview_details[elaboration_key_standard] = (
                        f"Overall world overview updated in Chapter {chapter_number}."
                    )
                try:
                    overview_item = WorldItem.from_dict(
                        category_name_llm,
                        "_overview_",
                        processed_overview_details,
                    )
                    results.setdefault(category_name_llm, {})["_overview_"] = (
                        overview_item
                    )
                except Exception as e:
                    logger.error(
                        "Error creating WorldItem for overview in category '%s': %s",
                        category_name_llm,
                        e,
                        exc_info=True,
                    )
        else:
            if all(not isinstance(v, dict) for v in items_llm.values()):
                items_llm = {category_name_llm: items_llm}
            for item_name_llm, item_attributes_llm in items_llm.items():
                if not item_name_llm or not isinstance(item_attributes_llm, dict):
                    logger.warning(
                        "Skipping item with invalid name or attributes in category '%s': Name='%s'",
                        category_name_llm,
                        item_name_llm,
                    )
                    continue

                processed_item_details = _normalize_attributes(
                    item_attributes_llm,
                    WORLD_UPDATE_DETAIL_KEY_MAP,
                    WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
                )
                has_other_meaningful_item_attrs = any(
                    k not in ["modification_proposal", elaboration_key_standard] and v
                    for k, v in processed_item_details.items()
                )
                if (
                    not processed_item_details.get(elaboration_key_standard)
                    and has_other_meaningful_item_attrs
                ):
                    processed_item_details[elaboration_key_standard] = (
                        f"Item '{item_name_llm}' in category '{category_name_llm}' updated in Chapter {chapter_number}."
                    )
                try:
                    world_item_instance = WorldItem.from_dict(
                        category_name_llm,
                        item_name_llm,
                        processed_item_details,
                    )
                    category_dict_by_item_name[world_item_instance.name] = (
                        world_item_instance
                    )
                except Exception as e:
                    logger.error(
                        "Error creating WorldItem for '%s' in category '%s': %s",
                        item_name_llm,
                        category_name_llm,
                        e,
                        exc_info=True,
                    )
            if category_dict_by_item_name:
                results[category_name_llm] = category_dict_by_item_name
    return results


def parse_problem_list(text: str, category: str | None = None) -> list[ProblemDetail]:
    """Parse a JSON list of problem details."""
    if not text or not text.strip():
        return []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "status" in data and "no significant" in str(data["status"]).lower():
                return []
            if "problems" in data and isinstance(data["problems"], list):
                data = data["problems"]
        if not isinstance(data, list):
            raise ValueError("LLM output was not a list of problems")
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode JSON: %s", exc)
        return [
            ProblemDetail(
                issue_category=category or "meta",
                problem_description=f"Invalid JSON from LLM: {exc}",
                quote_from_original_text="N/A - Invalid JSON",
                quote_char_start=None,
                quote_char_end=None,
                sentence_char_start=None,
                sentence_char_end=None,
                suggested_fix_focus="Ensure LLM outputs valid JSON.",
            )
        ]

    problems: list[ProblemDetail] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning("Problem item is not a dict: %s", item)
            continue
        prob = ProblemDetail(
            issue_category=item.get("issue_category", category or "meta"),
            problem_description=item.get(
                "problem_description", "N/A - Missing description"
            ),
            quote_from_original_text=item.get(
                "quote_from_original_text", "N/A - General Issue"
            ),
            quote_char_start=None,
            quote_char_end=None,
            sentence_char_start=None,
            sentence_char_end=None,
            suggested_fix_focus=item.get(
                "suggested_fix_focus", "N/A - Missing suggestion"
            ),
            rewrite_instruction=item.get("rewrite_instruction"),
            severity=item.get("severity"),
            related_spans=item.get("related_spans"),
        )
        if category:
            prob.issue_category = category
        problems.append(prob)
    return problems
