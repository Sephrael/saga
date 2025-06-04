# utils/parsing_utils.py
import re
import logging
from typing import List, Dict, Any, Optional, Union, Pattern, Callable, Tuple

logger = logging.getLogger(__name__)

class ParseError(Exception):
    """Custom exception for parsing errors."""
    pass

# Default regex for splitting blocks (e.g., by "---")
DEFAULT_BLOCK_SEPARATOR_REGEX = r'\n\s*---\s*\n'
# MODIFIED: To handle optional bolding around keys
DEFAULT_KEY_VALUE_PATTERN = re.compile(
    r"^\s*(?:\*\*)?([A-Za-z0-9\s_()'.\"\-]+?)(?:\*\*)?:\s*(.*)$"
)
DEFAULT_LIST_ITEM_PREFIXES = ["- ", "* "]


def split_text_into_blocks(
    text: str,
    separator_regex_str: str = DEFAULT_BLOCK_SEPARATOR_REGEX,
    flags: int = re.MULTILINE
) -> List[str]:
    if not text or not text.strip():
        return []
    blocks = re.split(separator_regex_str, text.strip(), flags=flags)
    return [block.strip() for block in blocks if block and block.strip()]


def parse_key_value_block(
    block_text_or_lines: Union[str, List[str]],
    key_map: Dict[str, Union[str, Callable[[re.Match[str]], str]]],
    list_internal_keys: List[str],
    list_item_prefixes: Optional[List[str]] = None,
    key_value_pattern: Optional[Pattern[str]] = None,
    special_list_handling: Optional[Dict[str, Dict[str, Any]]] = None,
    allow_unknown_keys: bool = False,
    default_key_for_unmatched_lines: Optional[str] = None
) -> Dict[str, Any]:
    if list_item_prefixes is None: list_item_prefixes = DEFAULT_LIST_ITEM_PREFIXES
    if key_value_pattern is None: key_value_pattern = DEFAULT_KEY_VALUE_PATTERN
    if special_list_handling is None: special_list_handling = {}

    parsed_data: Dict[str, Any] = {}
    if default_key_for_unmatched_lines and default_key_for_unmatched_lines not in parsed_data:
        # Initialize as empty list if it's meant to collect unmatched lines
        if default_key_for_unmatched_lines in list_internal_keys or \
           (special_list_handling and default_key_for_unmatched_lines in special_list_handling and special_list_handling[default_key_for_unmatched_lines].get("type") == "list"):
            parsed_data[default_key_for_unmatched_lines] = []
        # else, it could be a single string field, but the typical use for default_key_for_unmatched_lines is collecting multiple lines.
        # For safety, let's assume it's a list if not specified.
        elif default_key_for_unmatched_lines not in key_map: # if it's not a known single field
             parsed_data[default_key_for_unmatched_lines] = []


    active_list_internal_key: Optional[str] = None
    active_list_values: List[str] = []
    lines = block_text_or_lines.splitlines() if isinstance(block_text_or_lines, str) else block_text_or_lines

    for line_num, line_text_original in enumerate(lines):
        line_text_for_default_unmatched = line_text_original
        line_for_processing = line_text_original.strip()

        if not line_for_processing:
            if active_list_internal_key:
                # Append empty string to represent a blank line within a list item's continuation
                # if the list item itself can be multi-line.
                # For simple lists of strings, this might not be desired.
                # Let's assume simple lists, so skip appending "" for now unless special handling dictates it.
                pass
            continue

        current_line_is_list_item_syntax = any(line_for_processing.startswith(p) for p in list_item_prefixes)

        content_to_match_kv = line_for_processing
        # If current line starts with a list item marker, we should try to parse the content *after* the marker as a K-V pair
        # ONLY if that list item itself is expected to be a dictionary or complex structure.
        # For simple lists of strings, we just capture the content.
        if current_line_is_list_item_syntax and active_list_internal_key:
            prefix_len_to_strip = next((len(p) for p in list_item_prefixes if line_for_processing.startswith(p)), 0)
            content_to_match_kv = line_for_processing[prefix_len_to_strip:].lstrip()
        # Otherwise, if not an active list OR it is a list item but not part of an active list value collection, use the whole stripped line.

        kv_match = key_value_pattern.match(line_for_processing) # Try matching the whole line first

        # Logic to finalize a list if a new key is encountered
        if active_list_internal_key and kv_match:
            # Check if the new key is genuinely a new field and not part of the list item's content
            key_from_llm_raw_new = kv_match.group(1).strip()
            normalized_key_from_llm_new = key_from_llm_raw_new.lower().replace(" ", "_")
            is_new_top_level_key = False
            if normalized_key_from_llm_new in key_map:
                is_new_top_level_key = True
            else:
                for pattern_or_str_key_map, _ in key_map.items():
                    if isinstance(pattern_or_str_key_map, re.Pattern) and pattern_or_str_key_map.match(key_from_llm_raw_new):
                        is_new_top_level_key = True
                        break
            
            # If the current line is a list item syntax, and it's a new key, this list ends.
            # If it's NOT list item syntax but a new key, this list also ends.
            if is_new_top_level_key:
                 if active_list_internal_key in parsed_data and isinstance(parsed_data[active_list_internal_key], list):
                     parsed_data[active_list_internal_key].extend(active_list_values)
                 else:
                     parsed_data[active_list_internal_key] = active_list_values
                 active_list_internal_key = None
                 active_list_values = []


        if kv_match:
            key_from_llm_raw = kv_match.group(1).strip()
            value_from_llm = kv_match.group(2).strip()
            normalized_key_from_llm = key_from_llm_raw.lower().replace(" ", "_")
            internal_key: Optional[str] = None

            if normalized_key_from_llm in key_map and isinstance(key_map[normalized_key_from_llm], str):
                internal_key = key_map[normalized_key_from_llm] # type: ignore
            else: # Check regex keys in key_map
                for pattern_or_str_key, internal_target_or_func in key_map.items():
                    if isinstance(pattern_or_str_key, re.Pattern):
                        regex_match_obj = pattern_or_str_key.match(key_from_llm_raw) # Match against raw key
                        if regex_match_obj:
                            if callable(internal_target_or_func):
                                internal_key = internal_target_or_func(regex_match_obj)
                            elif isinstance(internal_target_or_func, str):
                                internal_key = internal_target_or_func
                            break # First regex match wins

            if internal_key:
                if internal_key in list_internal_keys:
                    active_list_internal_key = internal_key
                    # Initialize list if not already (can happen if key appears multiple times)
                    if not isinstance(parsed_data.get(internal_key), list):
                        parsed_data[internal_key] = []
                    
                    special_handling_for_list = special_list_handling.get(internal_key)
                    if special_handling_for_list and "separator" in special_handling_for_list and value_from_llm:
                        # Value on the key line is itself a separated list
                        parsed_data[internal_key].extend([v.strip() for v in value_from_llm.split(special_handling_for_list["separator"]) if v.strip()])
                    elif value_from_llm: # Value on key line is the first item of the list
                        active_list_values = [value_from_llm] # Start new list collection
                    else: # No value on key line, expect subsequent list items
                        active_list_values = []
                else: # Not a list key, direct assignment
                    if internal_key == "scene_number": # Special handling for scene_number
                        try: parsed_data[internal_key] = int(value_from_llm)
                        except ValueError:
                            logger.warning(f"Parser: Invalid int value for '{internal_key}': '{value_from_llm}'. Storing as string.")
                            parsed_data[internal_key] = value_from_llm
                    else:
                        parsed_data[internal_key] = value_from_llm
            elif default_key_for_unmatched_lines and not allow_unknown_keys:
                if isinstance(parsed_data.get(default_key_for_unmatched_lines), list):
                    parsed_data[default_key_for_unmatched_lines].append(line_text_for_default_unmatched)
                else: # If it was not initialized as a list (e.g. not in list_internal_keys)
                    parsed_data[default_key_for_unmatched_lines] = [line_text_for_default_unmatched]

            elif allow_unknown_keys:
                 parsed_data[key_from_llm_raw] = value_from_llm
                 logger.debug(f"Parser: Stored unknown key '{key_from_llm_raw}' because allow_unknown_keys is True.")

        elif current_line_is_list_item_syntax and active_list_internal_key:
            prefix_len_to_strip = next((len(p) for p in list_item_prefixes if line_for_processing.startswith(p)), 0)
            list_item_content = line_for_processing[prefix_len_to_strip:].strip()
            if list_item_content: # Only add non-empty items
                active_list_values.append(list_item_content)

        elif active_list_internal_key: # Line is not a K-V, not a list item, but a list is active (continuation)
            if line_for_processing: # If it's a non-empty continuation line
                active_list_values[-1] = active_list_values[-1] + "\n" + line_text_original # Append with original indent
        
        elif default_key_for_unmatched_lines:
             if isinstance(parsed_data.get(default_key_for_unmatched_lines), list):
                 parsed_data[default_key_for_unmatched_lines].append(line_text_for_default_unmatched)
             else: # If it was not initialized as a list
                 parsed_data[default_key_for_unmatched_lines] = [line_text_for_default_unmatched]


    # Finalize any pending active list
    if active_list_internal_key:
        if active_list_internal_key in parsed_data and isinstance(parsed_data[active_list_internal_key], list):
            parsed_data[active_list_internal_key].extend(active_list_values)
        else: # If the key was just declared but had no items on its line or subsequent lines
            parsed_data[active_list_internal_key] = active_list_values
    
    # Ensure all list_internal_keys are present and are lists
    for l_key in list_internal_keys:
        if l_key in parsed_data:
            if not isinstance(parsed_data[l_key], list):
                # If it was assigned a single string value (e.g. from a key-value pair where value was not empty)
                # and it's supposed to be a list, convert it to a list with one item.
                if parsed_data[l_key] is not None and str(parsed_data[l_key]).strip():
                    parsed_data[l_key] = [str(parsed_data[l_key])]
                else: # If it's empty or None, make it an empty list
                    parsed_data[l_key] = []
        else: # If key is missing entirely
            parsed_data[l_key] = []

    return parsed_data

_HIERARCHICAL_STRICT_WORLD_CATEGORIES = ["Overview", "Locations", "Factions", "Systems", "Lore", "History", "Society"]
_COMPILED_CATEGORY_ALTERNATION = "|".join(cat for cat in _HIERARCHICAL_STRICT_WORLD_CATEGORIES)

WORLD_CATEGORY_HEADER_PATTERN = re.compile(
   r"^\s*(?:Category\s*:\s*)?(?:\*\*)?(" + _COMPILED_CATEGORY_ALTERNATION + r")(?:\*\*)?\s*:?\s*$",
   re.IGNORECASE | re.MULTILINE | re.UNICODE
)

SIMPLER_WORLD_CATEGORY_HEADER_PATTERN = re.compile(
   r"^\s*\*\*(Overview|Locations|Factions|Systems|Lore|History|Society)\*\*\s*:\s*$",
   re.IGNORECASE | re.UNICODE
)

WORLD_ITEM_HEADER_PATTERN = re.compile(
    r"^\s*(?:Item\s*:\s*)?(?:\*\*)?([A-Za-z0-9\s_()'.\"\-]+?)(?:\*\*)?\s*:?\s*(.*)$"
)
WORLD_ITEM_HEADER_PATTERN_NO_COLON_EOL = re.compile(
    r"^\s*(?:Item\s*:\s*)?(?:\*\*)?([A-Za-z0-9\s_()'.\"\-]+?)(?:\*\*)?\s*:?\s*$"
)

def parse_hierarchical_structured_text(
    text_block: str,
    category_pattern: Pattern[str],
    item_pattern_with_content: Pattern[str],
    item_pattern_name_only: Pattern[str],
    detail_key_map: Dict[str, Union[str, Callable[[re.Match[str]], str]]],
    detail_list_internal_keys: List[str],
    overview_category_internal_key: Optional[str] = "_overview_",
    detail_list_item_prefixes: Optional[List[str]] = None,
    detail_key_value_pattern: Optional[Pattern[str]] = None
) -> Dict[str, Any]:
    parsed_hier_data: Dict[str, Any] = {}
    if not text_block.strip():
        return parsed_hier_data

    lines = text_block.splitlines()

    current_category_llm_raw: Optional[str] = None
    current_category_internal: Optional[str] = None
    current_item_name: Optional[str] = None
    current_item_detail_lines: List[str] = []

    effective_detail_kv_pattern = detail_key_value_pattern if detail_key_value_pattern is not None else DEFAULT_KEY_VALUE_PATTERN

    def _finalize_current_item_or_overview_details():
        nonlocal current_category_internal, current_item_name, current_item_detail_lines, current_category_llm_raw
        if not current_category_internal:
            if current_item_detail_lines:
                 logger.debug(f"HParser: Orphaned detail lines found without active category: {current_item_detail_lines[:2]}")
            current_item_detail_lines = []
            current_item_name = None
            return

        if current_item_detail_lines:
            item_details_dict = parse_key_value_block(
                current_item_detail_lines,
                detail_key_map,
                detail_list_internal_keys,
                list_item_prefixes=detail_list_item_prefixes,
                key_value_pattern=effective_detail_kv_pattern
            )

            target_dict_for_category = parsed_hier_data.setdefault(current_category_internal, {})
            if not isinstance(target_dict_for_category, dict):
                logger.error(f"HParser: Target for category '{current_category_internal}' is not a dict. Forcing to dict.")
                target_dict_for_category = {}
                parsed_hier_data[current_category_internal] = target_dict_for_category

            if item_details_dict:
                if current_item_name:
                    target_dict_for_category[current_item_name] = item_details_dict
                elif current_category_internal == overview_category_internal_key:
                    target_dict_for_category.update(item_details_dict)
                else:
                    logger.warning(f"HParser: Finalizing details for category '{current_category_internal}' but no current_item_name and not overview. Details: {item_details_dict}")

        current_item_name = None
        current_item_detail_lines = []

    for line_num, line_text_original_case in enumerate(lines):
        line_stripped = line_text_original_case.strip()

        category_match = category_pattern.match(line_stripped)
        if category_match:
            matched_category_group = category_match.group(1)

            _finalize_current_item_or_overview_details()
            current_category_llm_raw = matched_category_group.strip()
            current_category_internal = current_category_llm_raw.lower().replace(" ", "_")
            logger.debug(f"HParser: Line {line_num+1}: Switched to CATEGORY '{current_category_llm_raw}' (Internal: '{current_category_internal}')")
            parsed_hier_data.setdefault(current_category_internal, {})
            continue

        if not current_category_internal:
            if line_stripped: # Content before any category header
                logger.debug(f"HParser: Line {line_num+1}: Ignoring pre-category content: '{line_stripped}'")
            continue

        is_overview_cat = (current_category_internal == overview_category_internal_key)
        line_processed_as_item_header = False

        if not is_overview_cat:
            potential_item_name = None
            content_as_detail_from_item_line = None

            item_match_wc = item_pattern_with_content.match(line_text_original_case) # Match on original to preserve leading spaces for detail lines
            if item_match_wc:
                potential_item_name = item_match_wc.group(1).strip()
                content_as_detail_from_item_line = item_match_wc.group(2).strip() # This is content on the same line as item name
            else:
                item_match_no = item_pattern_name_only.match(line_stripped)
                if item_match_no:
                    potential_item_name = item_match_no.group(1).strip()

            if potential_item_name:
                # Check if this "item name" is actually a detail key for the current_item_name (if any)
                # This avoids treating "Description: Bla" under "Item: Foo" as a new item "Description"
                normalized_potential_item_as_detail_key = potential_item_name.lower().replace(" ", "_")
                is_actually_a_detail_key = False
                if detail_key_map.get(normalized_potential_item_as_detail_key) or \
                   any(isinstance(k_map, re.Pattern) and k_map.match(potential_item_name) for k_map in detail_key_map.keys()):
                    is_actually_a_detail_key = True

                if not is_actually_a_detail_key:
                    _finalize_current_item_or_overview_details()
                    current_item_name = potential_item_name
                    current_item_detail_lines = [] # Reset for new item
                    logger.debug(f"HParser: Line {line_num+1}: Switched to ITEM '{current_item_name}' in category '{current_category_internal}'")

                    if content_as_detail_from_item_line:
                        # This line IS the item header AND has content for it
                        # Need to pass it to parse_key_value_block as if it's a detail line.
                        # This requires constructing a fake "key: value" string for that content.
                        # For now, we'll just add it to current_item_detail_lines if it's not empty.
                        # The key would be implicit. This part is tricky.
                        # Let's assume the content_as_detail_from_item_line is the "description" if no explicit key is there.
                        # A simpler approach: if item_pattern_with_content matched, the group(2) is the first line of details.
                        current_item_detail_lines.append(item_match_wc.group(1).strip() + ": " + content_as_detail_from_item_line) # Reconstruct as a K:V line for parsing

                    line_processed_as_item_header = True

        if line_processed_as_item_header:
            continue

        if current_category_internal: # Add line to current item's details or overview's details
            current_item_detail_lines.append(line_text_original_case)

    _finalize_current_item_or_overview_details()
    return parsed_hier_data

# --- KG Triple Parsing ---
KG_LIST_FORMAT_PATTERN = re.compile(
    r"^\s*-\s*\[\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*,\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*,\s*['\"]?([^,'\"\[\]]+?)['\"]?\s*\]\s*$",
    re.MULTILINE
)
KG_KV_FORMAT_PATTERN = re.compile(
    r"^\s*Subject:\s*(.+?)\s*[,;]\s*Predicate:\s*(.+?)\s*[,;]\s*Object:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE
)
KG_PIPE_FORMAT_PATTERN = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$", re.MULTILINE)

# New: Pattern to capture EntityType:EntityName
ENTITY_TYPE_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_]+?)\s*:\s*(.+?)\s*$")

def _parse_entity_type_and_name(entity_str: str) -> Dict[str, str]:
    """Parses 'EntityType:EntityName' or just 'EntityName' into a dict."""
    entity_str_cleaned = entity_str.strip()
    match = ENTITY_TYPE_NAME_PATTERN.match(entity_str_cleaned)
    if match:
        return {"type": match.group(1).strip(), "name": match.group(2).strip()}
    return {"type": None, "name": entity_str_cleaned} # Default type to None if not specified

def _parse_triple_from_parts(s_raw: str, p_raw: str, o_raw: str) -> Optional[Dict[str, Any]]:
    s_cleaned = re.sub(r"^['\"\[\(]+|['\"\]\)]+$", "", s_raw.strip()).strip()
    p_cleaned = re.sub(r"^['\"\[\(]+|['\"\]\)]+$", "", p_raw.strip()).strip()
    o_cleaned = re.sub(r"^['\"\[\(]+|['\"\]\)]+$", "", o_raw.strip()).strip()

    if not (s_cleaned and p_cleaned and o_cleaned):
        return None

    subject_info = _parse_entity_type_and_name(s_cleaned)
    predicate_info = p_cleaned # Predicate is just a string

    # Check if object is a literal (e.g., quoted string, number, boolean)
    # This is a heuristic. A more robust way might involve checking if o_cleaned
    # matches any known entity types or if it's explicitly marked as literal by LLM.
    # For now, if it doesn't have an EntityType: prefix, and isn't a boolean, assume it could be literal or an untyped entity.
    # The add_kg_triples_batch_to_db will make the final call based on ValueNode or Entity node creation.

    object_parts = _parse_entity_type_and_name(o_cleaned)
    is_literal_object = False
    object_entity_payload = None
    object_literal_payload = None

    if object_parts["type"] is None: # No explicit type, could be literal or untyped entity name
        # Simple heuristic: if it's a number or "true"/"false" or starts/ends with quotes, treat as literal.
        if (re.match(r"^-?\d+(\.\d+)?$", o_cleaned)) or \
           (o_cleaned.lower() in ["true", "false"]) or \
           ((o_cleaned.startswith('"') and o_cleaned.endswith('"')) or \
            (o_cleaned.startswith("'") and o_cleaned.endswith("'"))):
            is_literal_object = True
            object_literal_payload = o_cleaned.strip('"\'') # Store unquoted for numbers/bools
        else: # Assume it's an entity name without a type
            object_entity_payload = {"name": object_parts["name"], "type": None}
    else: # Explicit type provided
        object_entity_payload = object_parts


    return {
        "subject": subject_info,
        "predicate": predicate_info,
        "object_entity": object_entity_payload,
        "object_literal": object_literal_payload,
        "is_literal_object": is_literal_object
    }

def _parse_triple_list_format_structured(line: str) -> Optional[Dict[str, Any]]:
    match = KG_LIST_FORMAT_PATTERN.match(line)
    if match:
        s, p, o = match.groups()
        return _parse_triple_from_parts(s, p, o)
    return None

def _parse_triple_kv_format_structured(line: str) -> Optional[Dict[str, Any]]:
    match = KG_KV_FORMAT_PATTERN.match(line)
    if match:
        s, p, o = match.groups()
        return _parse_triple_from_parts(s,p,o)
    return None

def _parse_triple_pipe_format_structured(line: str) -> Optional[Dict[str, Any]]:
    match = KG_PIPE_FORMAT_PATTERN.match(line)
    if match:
        s,p,o = match.groups()
        return _parse_triple_from_parts(s,p,o)
    return None

def _parse_triple_comma_format_structured(line: str) -> Optional[Dict[str, Any]]:
    if line.startswith("[") and line.endswith("]"): return None
    if line.lower().startswith("subject:") : return None # Handled by KV format
    parts_raw = [part.strip() for part in line.split(',')]
    if len(parts_raw) == 3 and all(parts_raw):
        # Basic heuristic to avoid overly long "names" that are likely sentences
        if any(p.count(' ') > 7 for p in parts_raw):
             # Allow more spaces if an entity type is detected, as type names can be multi-word
            subject_has_type = ENTITY_TYPE_NAME_PATTERN.match(parts_raw[0])
            object_has_type = ENTITY_TYPE_NAME_PATTERN.match(parts_raw[2])
            if not (subject_has_type and parts_raw[0].count(' ') < 10) and \
               not (object_has_type and parts_raw[2].count(' ') < 10):
                # If neither subject nor object seem to have explicit types and are long, skip
                return None
        return _parse_triple_from_parts(parts_raw[0], parts_raw[1], parts_raw[2])
    return None

KG_TRIPLE_PARSING_STRATEGIES_STRUCTURED: List[Callable[[str], Optional[Dict[str, Any]]]] = [
    _parse_triple_list_format_structured,
    _parse_triple_kv_format_structured,
    _parse_triple_pipe_format_structured,
    _parse_triple_comma_format_structured, # Comma format is most ambiguous, try last
]

def parse_kg_triples_from_text(text_block: str) -> List[Dict[str, Any]]:
    """
    Parses a block of text for KG triples and returns them in a structured format.
    Expected input format per line:
    - SubjectEntityType:SubjectName | Predicate | ObjectEntityType:ObjectName
    - SubjectEntityType:SubjectName | Predicate | LiteralValue
    - Subject | Predicate | Object (where types might be inferred or are general)
    - Variations like [S, P, O] or Subject: S, Predicate: P, Object: O
    Returns a list of dictionaries, each representing a triple with keys:
    'subject': {'name': str, 'type': Optional[str]}
    'predicate': str
    'object_entity': Optional[{'name': str, 'type': Optional[str]}]
    'object_literal': Optional[str]
    'is_literal_object': bool
    """
    triples: List[Dict[str, Any]] = []
    for line_content in text_block.splitlines():
        line = line_content.strip()
        if not line: continue

        parsed_triple_dict: Optional[Dict[str, Any]] = None
        for strategy in KG_TRIPLE_PARSING_STRATEGIES_STRUCTURED:
            result = strategy(line)
            if result:
                parsed_triple_dict = result
                break
        
        if parsed_triple_dict:
            # Validate essential parts of the parsed dict
            subject_info = parsed_triple_dict.get("subject")
            predicate_str = parsed_triple_dict.get("predicate")
            obj_ent_info = parsed_triple_dict.get("object_entity")
            obj_lit_val = parsed_triple_dict.get("object_literal")

            valid_subject = isinstance(subject_info, dict) and subject_info.get("name")
            valid_predicate = isinstance(predicate_str, str) and predicate_str.strip()
            valid_object = (isinstance(obj_ent_info, dict) and obj_ent_info.get("name")) or \
                           (parsed_triple_dict.get("is_literal_object") and obj_lit_val is not None) # obj_lit_val can be empty string ""

            if valid_subject and valid_predicate and valid_object:
                triples.append(parsed_triple_dict)
            else:
                logger.warning(f"KG triple parsing: Skipped triple due to invalid structure after parsing line: '{line}' -> Parsed: {parsed_triple_dict}")
        elif line and not line.lower().startswith("###"): # Avoid logging for section headers
            logger.debug(f"KG triple parsing: Could not parse line into a structured triple: '{line}'")

    if not triples and text_block.strip() and not text_block.lower().startswith(("no kg triples", "none", "###")):
        logger.warning(f"KG triple parsing: No structured triples extracted from non-empty text block: '{text_block[:200].replace(chr(10), ' ')}...'")
    return triples