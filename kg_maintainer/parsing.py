# kg_maintainer/parsing.py
import json  # Added json
import logging  # Added logging
from typing import Any, Dict, List  # Added Any, List

from .models import CharacterProfile, WorldItem

logger = logging.getLogger(__name__)

CHAR_UPDATE_KEY_MAP = {
    "desc": "description",
    "description": "description",
    "traits": "traits",
    "relationships": "relationships",
    "status": "status",
    "modification proposal": "modification_proposal",
    # Add other keys LLM might produce, mapping to CharacterProfile fields
    # e.g. "aliases": "aliases" if LLM provides aliases as a list
}

CHAR_UPDATE_LIST_INTERNAL_KEYS = [
    "traits",
    "relationships",
    "aliases",
]  # Added aliases as example

# No longer need CHAR_UPDATE_PKVB_SPECIAL_HANDLING as
# parse_key_value_block is removed

WORLD_UPDATE_DETAIL_KEY_MAP = {
    # Ensure these keys match what LLM will produce in JSON
    "desc": "description",
    "description": "description",
    "atmosphere": "atmosphere",  # Added from original example
    "goals": "goals",
    "rules": "rules",
    "key elements": "key_elements",
    "traits": "traits",  # Ensure this is a list if LLM provides a string
    "modification proposal": "modification_proposal",
}
WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS = [
    "goals",
    "rules",
    "key_elements",
    "traits",
]  # Ensure these are lists


def _normalize_attributes(
    attributes_dict: Dict[str, Any],
    key_map: Dict[str, str],
    list_keys: List[str],
) -> Dict[str, Any]:
    normalized_attrs: Dict[str, Any] = {}
    if not isinstance(attributes_dict, dict):
        logger.warning(
            "Input to _normalize_attributes was not a dict: %s",
            type(attributes_dict),
        )
        return {}

    for key, value in attributes_dict.items():
        # Normalize the key from LLM JSON for matching against key_map
        normalized_llm_key = key.lower().replace(" ", "_")
        mapped_key = key_map.get(
            normalized_llm_key, normalized_llm_key
        )  # Use normalized if not in map

        if mapped_key in list_keys:
            if isinstance(value, list):
                normalized_attrs[mapped_key] = value
            elif isinstance(value, dict):
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

    # Ensure all list_keys are present and are lists in the final output
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


def parse_unified_character_updates(
    json_text_block: str, chapter_number: int
) -> Dict[str, CharacterProfile]:
    """Parse character update JSON provided by LLM."""
    char_updates: Dict[str, CharacterProfile] = {}
    if not json_text_block.strip():
        return char_updates

    try:
        # LLM is expected to output a dict where keys are character names
        # and values are dicts of their attributes.
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
                "Skipping character with invalid name or attributes: Name='%s',"
                " Attrs_Type='%s'",
                char_name,
                type(char_attributes_llm),
            )
            continue

        # Normalize keys from LLM (e.g. "desc" to "description")
        # and ensure list types
        processed_char_attributes = _normalize_attributes(
            char_attributes_llm, CHAR_UPDATE_KEY_MAP, CHAR_UPDATE_LIST_INTERNAL_KEYS
        )

        # Handle relationships if they need structuring from list to dict
        # Assuming LLM provides relationships as a list of strings
        # like "Target: Detail" or just "Target"
        # Or ideally, as a dict: {"Target": "Detail"}
        rels_val = processed_char_attributes.get("relationships")
        if isinstance(rels_val, list):
            rels_list = rels_val
            rels_dict: Dict[str, str] = {}
            for rel_entry in rels_list:
                if isinstance(rel_entry, str):
                    if ":" in rel_entry:
                        parts = rel_entry.split(":", 1)
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            rels_dict[parts[0].strip()] = parts[1].strip()
                        elif parts[0].strip():  # If only name is there before colon
                            rels_dict[parts[0].strip()] = "related"
                    elif rel_entry.strip():  # No colon, just a name
                        rels_dict[rel_entry.strip()] = "related"
                elif isinstance(
                    rel_entry, dict
                ):  # If LLM sends [{"name": "X", "detail": "Y"}]
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
        else:  # Ensure it's always a dict
            processed_char_attributes["relationships"] = {}

        dev_key_standard = f"development_in_chapter_{chapter_number}"
        # If LLM includes this key (even with different casing/spacing), it will be normalized by _normalize_attributes
        # if dev_key_standard is in CHAR_UPDATE_KEY_MAP. For now, handle it explicitly.
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

        # Add default development note if no specific one and other attributes exist
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
                f"Error creating CharacterProfile for '{char_name}': {e}. Attributes: {processed_char_attributes}",
                exc_info=True,
            )

    return char_updates


def parse_unified_world_updates(
    json_text_block: str, chapter_number: int
) -> Dict[str, Dict[str, WorldItem]]:
    """Parse world update JSON provided by LLM."""
    world_updates: Dict[str, Dict[str, WorldItem]] = {}
    if not json_text_block.strip():
        return world_updates

    try:
        # LLM is expected to output a dict where keys are category display names (e.g., "Locations")
        # and values are dicts of item names to their attribute dicts.
        parsed_data = json.loads(json_text_block)
        if not isinstance(parsed_data, dict):
            logger.error(
                f"World updates JSON was not a dictionary. Received: {type(parsed_data)}"
            )
            return world_updates
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse world updates JSON: {e}. Input: {json_text_block[:500]}..."
        )
        return world_updates

    results: Dict[str, Dict[str, WorldItem]] = {}
    for category_name_llm, items_llm in parsed_data.items():
        if not isinstance(items_llm, dict):
            logger.warning(
                f"Skipping category '{category_name_llm}' as its content is not a dictionary of items."
            )
            continue

        # category_name_llm is e.g. "Locations", "Faction Alpha". This is used as the .category for WorldItem
        # The WorldItem model itself might normalize this for ID generation.

        category_dict_by_item_name: Dict[str, WorldItem] = {}
        elaboration_key_standard = f"elaboration_in_chapter_{chapter_number}"

        if (
            category_name_llm.lower() == "overview"
            or category_name_llm.lower() == "_overview_"
        ):
            # Overview is a single item, its details are directly in items_llm
            processed_overview_details = _normalize_attributes(
                items_llm,
                WORLD_UPDATE_DETAIL_KEY_MAP,
                WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
            )
            if any(k != "modification_proposal" for k in processed_overview_details):
                # check if any meaningful data
                # Add default elaboration if not present
                if not processed_overview_details.get(elaboration_key_standard):
                    processed_overview_details[elaboration_key_standard] = (
                        f"Overall world overview updated in Chapter {chapter_number}."
                    )
                try:
                    # For overview, item_name is fixed (e.g., "_overview_")
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
        else:  # Regular category with multiple items
            for item_name_llm, item_attributes_llm in items_llm.items():
                if not item_name_llm or not isinstance(item_attributes_llm, dict):
                    logger.warning(
                        "Skipping item with invalid name or attributes in "
                        "category '%s': Name='%s'",
                        category_name_llm,
                        item_name_llm,
                    )
                    continue

                processed_item_details = _normalize_attributes(
                    item_attributes_llm,
                    WORLD_UPDATE_DETAIL_KEY_MAP,
                    WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
                )

                # Add default elaboration if not present and other
                # attributes exist
                has_other_meaningful_item_attrs = any(
                    k
                    not in [
                        "modification_proposal",
                        elaboration_key_standard,
                    ]
                    and v
                    for k, v in processed_item_details.items()
                )
                if (
                    not processed_item_details.get(elaboration_key_standard)
                    and has_other_meaningful_item_attrs
                ):
                    processed_item_details[elaboration_key_standard] = (
                        f"Item '{item_name_llm}' in category '{category_name_llm}' "
                        f"updated in Chapter {chapter_number}."
                    )

                try:
                    # item_name_llm is the display name from JSON key.
                    # WorldItem stores this as .name and normalizes it for .id
                    # along with category_name_llm.
                    world_item_instance = WorldItem.from_dict(
                        category_name_llm,
                        item_name_llm,
                        processed_item_details,
                    )

                    # Store in this category's dictionary using the item's display name as key.
                    # If LLM provides duplicate item names within the same category, last one wins.
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
