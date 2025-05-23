# utils/parsing_utils.py
import re
import logging
from typing import List, Dict, Any, Optional, Union, Pattern

logger = logging.getLogger(__name__)

class ParseError(Exception):
    """Custom exception for parsing errors."""
    pass

# Default regex for splitting blocks (e.g., by "---")
DEFAULT_BLOCK_SEPARATOR_REGEX = r'\n\s*---\s*\n'
# Default regex for key-value lines (captures key, value)
# Permissive key characters: Alphanumeric, space, underscore, parentheses, apostrophe, hyphen
DEFAULT_KEY_VALUE_PATTERN = re.compile(r"^\s*([A-Za-z0-9\s_()'-]+?):\s*(.*)$")
DEFAULT_LIST_ITEM_PREFIXES = ["- ", "* "]


def split_text_into_blocks(
    text: str,
    separator_regex_str: str = DEFAULT_BLOCK_SEPARATOR_REGEX,
    flags: int = re.MULTILINE
) -> List[str]:
    """
    Splits text into blocks using a regex separator.
    Filters out empty blocks.
    """
    if not text or not text.strip():
        return []
    blocks = re.split(separator_regex_str, text.strip(), flags=flags)
    return [block.strip() for block in blocks if block and block.strip()]


def parse_key_value_block(
    block_text_or_lines: Union[str, List[str]],
    key_map: Dict[str, str],
    list_internal_keys: List[str],
    list_item_prefixes: Optional[List[str]] = None,
    key_value_pattern: Optional[Pattern[str]] = None,
    special_list_handling: Optional[Dict[str, Dict[str, Any]]] = None, # E.g. {"characters_involved": {"separator": ","}}
    allow_unknown_keys: bool = False,
    default_key_for_unmatched_lines: Optional[str] = None # If a line doesn't match K:V or list, append to this key's list
) -> Dict[str, Any]:
    """
    Parses a block of text (or lines) into a dictionary based on key_map.
    Handles simple key-value pairs and lists.

    Args:
        block_text_or_lines: The text block or list of lines to parse.
        key_map: Maps normalized LLM display keys (lowercase, underscores for spaces)
                 to internal canonical keys for the output dictionary.
        list_internal_keys: A list of *internal* keys that should result in list values.
        list_item_prefixes: Prefixes identifying list items (e.g., ["- ", "* "]).
        key_value_pattern: Compiled regex to identify key-value lines. Must capture (key, value).
        special_list_handling: Optional dict to define custom parsing for specific list keys.
                               e.g., {"internal_key_name": {"separator": ","}} for comma-separated lists on one line.
        allow_unknown_keys: If True, unmatched keys are logged but not added. If False (default), they are ignored.
                             (Note: current implementation logs and ignores)
        default_key_for_unmatched_lines: If a line is not a K:V, list item, or list header,
                                         it will be appended to a list under this internal key.
    Returns:
        A dictionary with internal keys and parsed values.
    """
    if list_item_prefixes is None: list_item_prefixes = DEFAULT_LIST_ITEM_PREFIXES
    if key_value_pattern is None: key_value_pattern = DEFAULT_KEY_VALUE_PATTERN
    if special_list_handling is None: special_list_handling = {}


    parsed_data: Dict[str, Any] = {}
    if default_key_for_unmatched_lines and default_key_for_unmatched_lines not in parsed_data:
        parsed_data[default_key_for_unmatched_lines] = []

    active_list_internal_key: Optional[str] = None
    active_list_values: List[str] = []

    lines = block_text_or_lines.splitlines() if isinstance(block_text_or_lines, str) else block_text_or_lines

    for line_num, line_text in enumerate(lines):
        line = line_text.strip()
        # Skip fully empty lines. Indented lines might be part of multi-line values (if supported).
        if not line_text.strip() and not line_text: # Distinguish between empty and "   "
            continue

        # Finalize active list if current line doesn't continue it and is not empty indented
        is_list_item_line = any(line.startswith(p) for p in list_item_prefixes)
        
        # If we have an active list, and this line is NOT a list item for it
        if active_list_internal_key and not is_list_item_line:
            # Check if this line STARTS a new known key or list header. If so, finalize.
            # Otherwise, it might be a multi-line part of the *last* list item (not supported by this simple parser)
            # or an unmatched line.
            potential_new_key_match = key_value_pattern.match(line)
            normalized_line_as_key_header = line.replace(":", "").strip().lower().replace(" ", "_")
            is_new_known_key = potential_new_key_match and key_map.get(potential_new_key_match.group(1).strip().lower().replace(" ", "_"))
            is_new_known_list_header = key_map.get(normalized_line_as_key_header) and key_map[normalized_line_as_key_header] in list_internal_keys

            if is_new_known_key or is_new_known_list_header or not line_text.startswith("  "): # Finalize if new key or not indented
                if active_list_internal_key in parsed_data:
                    parsed_data[active_list_internal_key].extend(active_list_values)
                else:
                    parsed_data[active_list_internal_key] = active_list_values
                active_list_internal_key = None
                active_list_values = []

        # Attempt to match "Key: Value"
        kv_match = key_value_pattern.match(line)
        if kv_match:
            key_from_llm_raw = kv_match.group(1).strip()
            value_from_llm = kv_match.group(2).strip()
            
            normalized_key_from_llm = key_from_llm_raw.lower().replace(" ", "_")
            internal_key = key_map.get(normalized_key_from_llm)

            if internal_key:
                if internal_key in list_internal_keys:
                    active_list_internal_key = internal_key
                    active_list_values = []
                    parsed_data[internal_key] = [] # Initialize list

                    special_handling = special_list_handling.get(internal_key)
                    if special_handling and "separator" in special_handling and value_from_llm:
                        # Custom separator for this list type (e.g., comma-separated characters_involved)
                        active_list_values.extend([v.strip() for v in value_from_llm.split(special_handling["separator"]) if v.strip()])
                    elif value_from_llm: # Content on the same line as list key
                        is_value_list_item = any(value_from_llm.startswith(p) for p in list_item_prefixes)
                        if is_value_list_item:
                            prefix_len = next((len(p) for p in list_item_prefixes if value_from_llm.startswith(p)), 0)
                            active_list_values.append(value_from_llm[prefix_len:].strip())
                        else: # Treat as a single item if not formatted as list item prefix
                            active_list_values.append(value_from_llm)
                else: # Simple key-value
                    # Potentially cast to int if specified in a more complex key_config (future)
                    if internal_key == "scene_number": # Example: special handling
                        try: parsed_data[internal_key] = int(value_from_llm)
                        except ValueError:
                            logger.warning(f"Parser: Invalid int value for '{internal_key}': '{value_from_llm}'. Storing as string.")
                            parsed_data[internal_key] = value_from_llm
                    else:
                        parsed_data[internal_key] = value_from_llm
            else:
                logger.debug(f"Parser: Unknown key '{key_from_llm_raw}' (normalized: '{normalized_key_from_llm}') in block.")
                if default_key_for_unmatched_lines:
                    parsed_data[default_key_for_unmatched_lines].append(line_text) # Add raw line
        
        # Handle list items for an active list key
        elif is_list_item_line and active_list_internal_key:
            prefix_len = next((len(p) for p in list_item_prefixes if line.startswith(p)), 0)
            active_list_values.append(line[prefix_len:].strip())
        
        # Handle list header on its own line (e.g., "Key Dialogue Points:")
        elif not is_list_item_line and not kv_match: # Not a K:V, not a list item
            normalized_line_as_key_header = line.replace(":", "").strip().lower().replace(" ", "_")
            internal_key_for_list_header = key_map.get(normalized_line_as_key_header)

            if internal_key_for_list_header and internal_key_for_list_header in list_internal_keys:
                active_list_internal_key = internal_key_for_list_header
                active_list_values = []
                parsed_data[internal_key_for_list_header] = [] # Initialize
            elif default_key_for_unmatched_lines:
                 parsed_data[default_key_for_unmatched_lines].append(line_text) # Add raw line
            # else: Line is not a known key, list item, or list header.

    # Finalize any list at the end of the block
    if active_list_internal_key:
        if active_list_internal_key in parsed_data:
            parsed_data[active_list_internal_key].extend(active_list_values)
        else:
            parsed_data[active_list_internal_key] = active_list_values
    
    # Ensure all list_internal_keys are indeed lists in the final dict
    for l_key in list_internal_keys:
        if l_key in parsed_data and not isinstance(parsed_data[l_key], list):
            # This can happen if a list key had a single non-prefixed value on its line
            # and no subsequent list items.
            logger.debug(f"Parser: Converting single value for list key '{l_key}' to list: {parsed_data[l_key]}")
            parsed_data[l_key] = [parsed_data[l_key]]
        elif l_key not in parsed_data: # Ensure list keys are present, even if empty
            parsed_data[l_key] = []

    return parsed_data


def parse_hierarchical_structured_text(
    text_block: str,
    category_pattern: Pattern[str],  # Regex to identify category. Group 1 must be cat name.
    item_pattern: Pattern[str],      # Regex to identify item. Group 1 must be item name.
    detail_key_map: Dict[str, str],
    detail_list_internal_keys: List[str],
    overview_category_internal_key: Optional[str] = "_overview_", # If a category should have its details parsed directly
    detail_list_item_prefixes: Optional[List[str]] = None,
    detail_key_value_pattern: Optional[Pattern[str]] = None
) -> Dict[str, Any]:
    """
    Parses text with a hierarchical structure: categories, items within categories,
    and key-value details for each item. Adapted from initial_setup_logic.
    """
    parsed_hier_data: Dict[str, Any] = {}
    lines = text_block.splitlines()

    current_category_internal: Optional[str] = None
    current_item_name: Optional[str] = None
    current_item_detail_lines: List[str] = []

    def _finalize_current_item_details():
        nonlocal current_category_internal, current_item_name, current_item_detail_lines
        if current_category_internal and current_item_detail_lines:
            item_details_dict = parse_key_value_block(
                current_item_detail_lines,
                detail_key_map,
                detail_list_internal_keys,
                list_item_prefixes=detail_list_item_prefixes,
                key_value_pattern=detail_key_value_pattern
            )
            if current_item_name: # Item within a regular category
                if current_category_internal not in parsed_hier_data:
                    parsed_hier_data[current_category_internal] = {}
                # Ensure the category entry is a dictionary
                if not isinstance(parsed_hier_data[current_category_internal], dict): 
                    logger.warning(f"HierarchicalParser: Category '{current_category_internal}' was not a dict. Resetting. Previous value: {parsed_hier_data[current_category_internal]}")
                    parsed_hier_data[current_category_internal] = {} 
                # Corrected line:
                parsed_hier_data[current_category_internal][current_item_name] = item_details_dict
            elif current_category_internal == overview_category_internal_key: # Details for an overview category
                if overview_category_internal_key not in parsed_hier_data:
                    parsed_hier_data[overview_category_internal_key] = {}
                # Ensure the overview category entry is a dictionary
                if not isinstance(parsed_hier_data[overview_category_internal_key], dict):
                    logger.warning(f"HierarchicalParser: Overview category '{overview_category_internal_key}' was not a dict. Resetting. Previous value: {parsed_hier_data[overview_category_internal_key]}")
                    parsed_hier_data[overview_category_internal_key] = {}
                parsed_hier_data[overview_category_internal_key].update(item_details_dict)
        
        current_item_name = None
        current_item_detail_lines = []

    for line_text in lines:
        line = line_text.strip()
        if not line: continue # Skip empty lines

        category_match = category_pattern.match(line)
        if category_match:
            _finalize_current_item_details() # Finalize any pending item from previous category
            
            cat_name_from_llm = category_match.group(1).strip()
            current_category_internal = cat_name_from_llm.lower().replace(" ", "_") # Example normalization
            
            if current_category_internal == overview_category_internal_key:
                current_item_name = None # Overview details are direct, not under items
            logger.debug(f"HierarchicalParser: Switched to category '{current_category_internal}'.")
            continue

        if not current_category_internal:
            logger.debug(f"HierarchicalParser: Skipping line, no active category: '{line}'")
            continue

        if current_category_internal != overview_category_internal_key:
            item_match = item_pattern.match(line)
            if item_match:
                _finalize_current_item_details() # Finalize previous item in the same category
                current_item_name = item_match.group(1).strip()
                current_item_detail_lines = []
                logger.debug(f"HierarchicalParser: New item '{current_item_name}' in category '{current_category_internal}'.")
                continue
        
        # If not a category or item header, it's a detail line
        current_item_detail_lines.append(line_text) # Pass raw line with original indentation

    _finalize_current_item_details() # Finalize the very last item
    return parsed_hier_data


def parse_kg_triples_from_text(text_block: str) -> List[List[str]]:
    """
    Parses KG triples from a text block.
    Supports various common LLM output formats for triples.
    """
    triples: List[List[str]] = []
    
    # Format: - [Subject, Predicate, Object] (with optional quotes)
    list_format_pattern = re.compile(
        r"^\s*-\s*\[\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*,\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*,\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*\]\s*$",
        re.MULTILINE
    )
    # Format: Subject: S, Predicate: P, Object: O (or using ';')
    key_value_format_pattern = re.compile(
        r"^\s*Subject:\s*(.+?)\s*[,;]\s*Predicate:\s*(.+?)\s*[,;]\s*Object:\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE
    )
    # Format: S | P | O
    pipe_format_pattern = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$", re.MULTILINE)

    for line_content in text_block.splitlines():
        line = line_content.strip()
        if not line: continue

        s, p, o = None, None, None
        
        list_match = list_format_pattern.match(line)
        if list_match: s, p, o = list_match.groups()
        else:
            kv_match = key_value_format_pattern.match(line)
            if kv_match: s, p, o = kv_match.groups()
            else:
                pipe_match = pipe_format_pattern.match(line)
                if pipe_match: s, p, o = pipe_match.groups()
                # Fallback for simple comma-separated, if exactly two commas and not empty parts
                elif line.count(',') == 2:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) == 3 and all(parts): # Ensure all parts are non-empty after strip
                        s, p, o = parts
        
        if s and p and o:
            s_cleaned, p_cleaned, o_cleaned = s.strip(), p.strip(), o.strip()
            # Further ensure they are not just quotes or brackets if those were part of the capture
            s_cleaned = re.sub(r"^['\"\[\]]+|['\"\[\]]+$", "", s_cleaned)
            p_cleaned = re.sub(r"^['\"\[\]]+|['\"\[\]]+$", "", p_cleaned)
            o_cleaned = re.sub(r"^['\"\[\]]+|['\"\[\]]+$", "", o_cleaned)

            if s_cleaned and p_cleaned and o_cleaned:
                triples.append([s_cleaned, p_cleaned, o_cleaned])
            else:
                logger.warning(f"KG triple parsing: Skipped triple due to empty component after cleaning from line: '{line}'")
        elif line: # Line was not empty but didn't match any pattern and wasn't parsed
            logger.debug(f"KG triple parsing: Could not parse line into a triple: '{line}'")
            
    if not triples and text_block.strip():
        logger.warning(f"KG triple parsing: No triples extracted from non-empty text block: '{text_block[:200]}...'")
        
    return triples