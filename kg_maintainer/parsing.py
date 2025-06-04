# kg_maintainer/parsing.py
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
    for category_name_from_parser, items in parsed_data.items():
        # Category name from parser (e.g. "Locations", "_overview_")
        # This becomes the 'category' argument for WorldItem.from_dict
        # The WorldItem model itself will normalize this for ID generation if needed,
        # but stores the passed category_name_from_parser as its .category attribute.
        
        elaboration_key_standard = f"elaboration_in_chapter_{chapter_number}"
        if category_name_from_parser == "_overview_":
            if items and isinstance(items, dict) and any(k != "modification_proposal" for k in items):
                specific_elab_key = next((k for k in items if k.lower() == elaboration_key_standard.lower()), None)
                if not specific_elab_key:
                    items[elaboration_key_standard] = (
                        f"Overall world overview mentioned or updated in Chapter {chapter_number}."
                    )
                elif specific_elab_key != elaboration_key_standard and items.get(specific_elab_key):
                    items[elaboration_key_standard] = items.pop(specific_elab_key)
                # For overview, item_name is fixed, category is fixed.
                results[category_name_from_parser] = {"_overview_": WorldItem.from_dict(category_name_from_parser, "_overview_", items)}
        elif isinstance(items, dict):
            cat_dict: Dict[str, WorldItem] = {}
            for item_name_from_parser, item_details in items.items():
                # item_name_from_parser is the display name like "The Red Key", "K"
                # This becomes the 'name' argument for WorldItem.from_dict
                # WorldItem model normalizes this for ID generation, stores original as .name
                if not item_name_from_parser or not isinstance(item_name_from_parser, str) or not item_name_from_parser.strip():
                    continue # Skip items with no valid name

                if isinstance(item_details, dict):
                    if any(k != "modification_proposal" for k in item_details):
                        specific_elab_key = next((k for k in item_details if k.lower() == elaboration_key_standard.lower()), None)
                        if not specific_elab_key:
                            item_details[elaboration_key_standard] = (
                                f"Item '{item_name_from_parser}' in category '{category_name_from_parser}' was mentioned or interacted with in Chapter {chapter_number}."
                            )
                        elif specific_elab_key != elaboration_key_standard and item_details.get(specific_elab_key):
                            item_details[elaboration_key_standard] = item_details.pop(specific_elab_key)
                    
                    try:
                        # Pass the category_name_from_parser and item_name_from_parser as they are.
                        # WorldItem.from_dict will handle ID generation based on normalized versions.
                        world_item_instance = WorldItem.from_dict(category_name_from_parser, item_name_from_parser, item_details)
                        # The key in cat_dict should be the canonical ID to prevent overwrites if LLM gives "K" and "k"
                        # which both normalize to the same WorldItem.id.
                        # If two items from LLM output result in the same WorldItem.id, the later one's details will
                        # overwrite the former's *at this parsing stage*. This merge is simple.
                        cat_dict[world_item_instance.id] = world_item_instance
                    except ValueError as e:
                        # This can happen if WorldItem.from_dict raises error for empty name/category
                        # (though we tried to pre-filter item_name_from_parser)
                        logger.error(f"Skipping world item due to validation error during WorldItem creation: {e}. Category: '{category_name_from_parser}', Item Name from LLM: '{item_name_from_parser}'")


            if cat_dict:
                # Convert cat_dict from {id: WorldItem} to {name: WorldItem} for consistency with old structure,
                # though this assumes names within a category (after WorldItem.from_dict name storage) are unique.
                # If two items had different IDs but ended up with same .name and .category, one would be lost here.
                # Better to keep cat_dict as {id: WorldItem} and let persist_world handle it.
                # For now, to minimize changes to merge.py, we'll keep the old structure,
                # but this could be a source of issues if two items get same ID but their original parsed names were different.
                
                # Let's change results to store by ID, and merge logic will need to adapt.
                # No, merge_world_item_updates expects Dict[str, Dict[str, WorldItem]] where inner key is name.
                # This means `parse_unified_world_updates` needs to ensure the keys of its *output* dictionary
                # (for a given category) are unique display names.
                # If two LLM items generate the same WorldItem.id (meaning they are the same logical item),
                # then `cat_dict[world_item_instance.id] = world_item_instance` already handles merging them
                # (last one wins).
                # The final `results[category_name_from_parser]` should then be built using `world_item_instance.name` as key.
                
                final_cat_dict_by_name: Dict[str, WorldItem] = {}
                for item_instance in cat_dict.values(): # cat_dict values are unique WorldItem instances by id
                    if item_instance.name in final_cat_dict_by_name:
                        logger.warning(f"During final structuring of parsed world items for category '{category_name_from_parser}', "
                                       f"duplicate display name '{item_instance.name}' encountered for different IDs. "
                                       f"ID {item_instance.id} will overwrite previous for this name. This implies an issue in LLM name consistency or parsing if IDs were meant to be different.")
                    final_cat_dict_by_name[item_instance.name] = item_instance
                
                if final_cat_dict_by_name:
                    results[category_name_from_parser] = final_cat_dict_by_name
    return results