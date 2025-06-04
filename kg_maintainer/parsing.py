"""Utilities for parsing text blocks into structured data for knowledge graph updates."""

from typing import Dict
import re
from parsing_utils import (
    parse_key_value_block,
    parse_hierarchical_structured_text,
    WORLD_CATEGORY_HEADER_PATTERN,
    WORLD_ITEM_HEADER_PATTERN,
    WORLD_ITEM_HEADER_PATTERN_NO_COLON_EOL,
)
from .models import CharacterProfile, WorldItem

CHAR_UPDATE_KEY_MAP = {
    "desc": "description",
    "description": "description",
    "traits": "traits",
    "relationships": "relationships",
    "status": "status",
    "modification proposal": "modification_proposal",
}

CHAR_UPDATE_LIST_INTERNAL_KEYS = ["traits", "relationships"]
CHAR_UPDATE_PKVB_SPECIAL_HANDLING = {
    "traits": "list",
    "relationships": "list",
}

WORLD_UPDATE_DETAIL_KEY_MAP = {
    "desc": "description",
    "description": "description",
    "goals": "goals",
    "rules": "rules",
    "key elements": "key_elements",
    "traits": "traits",
    "modification proposal": "modification_proposal",
}
WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS = ["goals", "rules", "key_elements", "traits"]


def parse_unified_character_updates(text_block: str, chapter_number: int) -> Dict[str, CharacterProfile]:
    """Parse character update blocks from text."""
    char_updates: Dict[str, CharacterProfile] = {}
    character_block_starts = list(re.finditer(r"^\s*Character:\s*(.+)$", text_block, re.IGNORECASE | re.MULTILINE))

    for i, start_match in enumerate(character_block_starts):
        char_name = start_match.group(1).strip()
        if not char_name:
            continue

        block_start_index = start_match.end()
        block_end_index = character_block_starts[i + 1].start() if i + 1 < len(character_block_starts) else len(text_block)
        individual_char_block_text = text_block[block_start_index:block_end_index].strip()

        if individual_char_block_text:
            parsed_char_data = parse_key_value_block(
                individual_char_block_text,
                CHAR_UPDATE_KEY_MAP,
                CHAR_UPDATE_LIST_INTERNAL_KEYS,
                special_list_handling=CHAR_UPDATE_PKVB_SPECIAL_HANDLING,
            )

            if "relationships" in parsed_char_data and isinstance(parsed_char_data["relationships"], list):
                rels_dict = {}
                for rel_str_or_item in parsed_char_data["relationships"]:
                    if isinstance(rel_str_or_item, str) and ":" in rel_str_or_item:
                        parts = rel_str_or_item.split(":", 1)
                        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                            rels_dict[parts[0].strip()] = parts[1].strip()
                    elif isinstance(rel_str_or_item, str) and rel_str_or_item.strip():
                        rels_dict[rel_str_or_item.strip()] = "related"
                parsed_char_data["relationships"] = rels_dict

            dev_key_standard = f"development_in_chapter_{chapter_number}"
            specific_dev_key_from_llm = next((k for k in parsed_char_data if k.lower() == dev_key_standard.lower()), None)

            if specific_dev_key_from_llm and specific_dev_key_from_llm != dev_key_standard:
                parsed_char_data[dev_key_standard] = parsed_char_data.pop(specific_dev_key_from_llm)
            elif not specific_dev_key_from_llm and any(k != "modification_proposal" for k in parsed_char_data):
                parsed_char_data[dev_key_standard] = f"Character '{char_name}' appeared or was mentioned in Chapter {chapter_number}."

            char_updates[char_name] = CharacterProfile.from_dict(char_name, parsed_char_data)
    return char_updates


def parse_unified_world_updates(text_block: str, chapter_number: int) -> Dict[str, Dict[str, WorldItem]]:
    """Parse world update blocks from text."""
    parsed_data = parse_hierarchical_structured_text(
        text_block,
        WORLD_CATEGORY_HEADER_PATTERN,
        WORLD_ITEM_HEADER_PATTERN,
        WORLD_ITEM_HEADER_PATTERN_NO_COLON_EOL,
        WORLD_UPDATE_DETAIL_KEY_MAP,
        WORLD_UPDATE_DETAIL_LIST_INTERNAL_KEYS,
        overview_category_internal_key="_overview_",
    )

    results: Dict[str, Dict[str, WorldItem]] = {}
    for category_name, items in parsed_data.items():
        elaboration_key_standard = f"elaboration_in_chapter_{chapter_number}"
        if category_name == "_overview_":
            if items and isinstance(items, dict) and any(k != "modification_proposal" for k in items):
                specific_elab_key = next((k for k in items if k.lower() == elaboration_key_standard.lower()), None)
                if not specific_elab_key:
                    items[elaboration_key_standard] = (
                        f"Overall world overview mentioned or updated in Chapter {chapter_number}."
                    )
                elif specific_elab_key != elaboration_key_standard and items.get(specific_elab_key):
                    items[elaboration_key_standard] = items.pop(specific_elab_key)
                results[category_name] = {"_overview_": WorldItem.from_dict(category_name, "_overview_", items)}
        elif isinstance(items, dict):
            cat_dict: Dict[str, WorldItem] = {}
            for item_name, item_details in items.items():
                if isinstance(item_details, dict):
                    if any(k != "modification_proposal" for k in item_details):
                        specific_elab_key = next((k for k in item_details if k.lower() == elaboration_key_standard.lower()), None)
                        if not specific_elab_key:
                            item_details[elaboration_key_standard] = (
                                f"Item '{item_name}' in category '{category_name}' was mentioned or interacted with in Chapter {chapter_number}."
                            )
                        elif specific_elab_key != elaboration_key_standard and item_details.get(specific_elab_key):
                            item_details[elaboration_key_standard] = item_details.pop(specific_elab_key)
                    cat_dict[item_name] = WorldItem.from_dict(category_name, item_name, item_details)
            if cat_dict:
                results[category_name] = cat_dict
    return results
