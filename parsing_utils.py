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
        parsed_data[default_key_for_unmatched_lines] = []

    active_list_internal_key: Optional[str] = None
    active_list_values: List[str] = []
    lines = block_text_or_lines.splitlines() if isinstance(block_text_or_lines, str) else block_text_or_lines

    for line_num, line_text_original in enumerate(lines):
        line_text_for_default_unmatched = line_text_original 
        line_for_processing = line_text_original.strip()

        if not line_for_processing: 
            if active_list_internal_key:
                active_list_values.append("") 
            continue

        current_line_is_list_item_syntax = any(line_for_processing.startswith(p) for p in list_item_prefixes)
        
        content_to_match_kv = line_for_processing
        if current_line_is_list_item_syntax:
            prefix_len_to_strip = next((len(p) for p in list_item_prefixes if line_for_processing.startswith(p)), 0)
            content_to_match_kv = line_for_processing[prefix_len_to_strip:].lstrip() 

        kv_match = key_value_pattern.match(content_to_match_kv)
        
        if active_list_internal_key and not current_line_is_list_item_syntax:
            is_new_key_on_original_line = False
            if not kv_match: 
                original_line_kv_match = key_value_pattern.match(line_for_processing) 
                if original_line_kv_match:
                    key_from_original_line = original_line_kv_match.group(1).strip().lower().replace(" ", "_")
                    if key_map.get(key_from_original_line) or \
                       any(isinstance(k_map, re.Pattern) and k_map.match(original_line_kv_match.group(1).strip()) for k_map in key_map.keys()):
                        is_new_key_on_original_line = True
            
            if is_new_key_on_original_line or (not line_text_original.startswith("  ") and not kv_match):
                if active_list_internal_key in parsed_data:
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
            else:
                for pattern_or_str_key, internal_target_or_func in key_map.items():
                    if isinstance(pattern_or_str_key, re.Pattern) and pattern_or_str_key.match(key_from_llm_raw):
                        if callable(internal_target_or_func):
                            match_obj = pattern_or_str_key.match(key_from_llm_raw)
                            if match_obj: internal_key = internal_target_or_func(match_obj)
                        elif isinstance(internal_target_or_func, str):
                            internal_key = internal_target_or_func
                        break
            
            if internal_key:
                if internal_key in list_internal_keys:
                    active_list_internal_key = internal_key
                    active_list_values = [] 
                    parsed_data[internal_key] = []
                    special_handling = special_list_handling.get(internal_key)
                    if special_handling and "separator" in special_handling and value_from_llm:
                        active_list_values.extend([v.strip() for v in value_from_llm.split(special_handling["separator"]) if v.strip()])
                    elif value_from_llm: 
                        is_value_list_item_formatted = any(value_from_llm.startswith(p) for p in list_item_prefixes)
                        if is_value_list_item_formatted:
                            val_prefix_len = next((len(p) for p in list_item_prefixes if value_from_llm.startswith(p)), 0)
                            active_list_values.append(value_from_llm[val_prefix_len:].strip())
                        else:
                            active_list_values.append(value_from_llm)
                else: 
                    if internal_key == "scene_number": 
                        try: parsed_data[internal_key] = int(value_from_llm)
                        except ValueError:
                            logger.warning(f"Parser: Invalid int value for '{internal_key}': '{value_from_llm}'. Storing as string.")
                            parsed_data[internal_key] = value_from_llm
                    else:
                        parsed_data[internal_key] = value_from_llm
            elif default_key_for_unmatched_lines and not allow_unknown_keys:
                parsed_data[default_key_for_unmatched_lines].append(line_text_for_default_unmatched)
            elif allow_unknown_keys:
                 parsed_data[key_from_llm_raw] = value_from_llm 
                 logger.debug(f"Parser: Stored unknown key '{key_from_llm_raw}' because allow_unknown_keys is True.")

        elif current_line_is_list_item_syntax and active_list_internal_key:
            active_list_values.append(content_to_match_kv) 
        
        elif default_key_for_unmatched_lines: 
             parsed_data[default_key_for_unmatched_lines].append(line_text_for_default_unmatched)

    if active_list_internal_key:
        if active_list_internal_key in parsed_data and isinstance(parsed_data[active_list_internal_key], list):
            parsed_data[active_list_internal_key].extend(active_list_values)
        else:
            parsed_data[active_list_internal_key] = active_list_values
    
    for l_key in list_internal_keys:
        if l_key in parsed_data and not isinstance(parsed_data[l_key], list):
            parsed_data[l_key] = [parsed_data[l_key]]
        elif l_key not in parsed_data:
            parsed_data[l_key] = []

    return parsed_data

_HIERARCHICAL_STRICT_WORLD_CATEGORIES = ["Overview", "Locations", "Factions", "Systems", "Lore", "History", "Society"]
_COMPILED_CATEGORY_ALTERNATION = "|".join(cat for cat in _HIERARCHICAL_STRICT_WORLD_CATEGORIES)

WORLD_CATEGORY_HEADER_PATTERN = re.compile(
   r"^\s*(?:Category\s*:\s*)?(?:\*\*)?(" + _COMPILED_CATEGORY_ALTERNATION + r")(?:\*\*)?\s*:\s*$",
   re.IGNORECASE | re.MULTILINE | re.UNICODE 
)

# Potential simplified pattern for testing if the main one mysteriously fails
# This makes bolding mandatory and removes the optional "Category: " prefix
SIMPLER_WORLD_CATEGORY_HEADER_PATTERN = re.compile(
   r"^\s*\*\*(Overview|Locations|Factions|Systems|Lore|History|Society)\*\*\s*:\s*$", # Note: No optional (?:Category...)? and (?:\*\*)? are now just \*\*
   re.IGNORECASE | re.UNICODE # Removed re.MULTILINE as it's applied per line
)

WORLD_ITEM_HEADER_PATTERN = re.compile(
    r"^\s*(?:\*\*)?([A-Za-z0-9\s_()'.\"\-]+?)(?:\*\*)?:\s*(.*)$"
)
WORLD_ITEM_HEADER_PATTERN_NO_COLON_EOL = re.compile(
    r"^\s*(?:\*\*)?([A-Za-z0-9\s_()'.\"\-]+?)(?:\*\*)?(?::\s*)?$"
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
                 logger.warning(f"HParser: Orphaned detail lines found without active category: {current_item_detail_lines[:2]}")
            current_item_detail_lines = [] 
            current_item_name = None       
            return

        if current_item_detail_lines: 
            item_or_overview_label = f"Item: '{current_item_name}'" if current_item_name else "Overview Details"
            
            item_details_dict = parse_key_value_block(
                current_item_detail_lines,
                detail_key_map,
                detail_list_internal_keys,
                list_item_prefixes=detail_list_item_prefixes,
                key_value_pattern=effective_detail_kv_pattern
            )
            
            target_dict_for_category = parsed_hier_data.setdefault(current_category_internal, {})
            if not isinstance(target_dict_for_category, dict): 
                logger.error(f"HParser: Target for category '{current_category_internal}' is not a dict after setdefault. This is unexpected. Forcing to dict.")
                target_dict_for_category = {}
                parsed_hier_data[current_category_internal] = target_dict_for_category

            if item_details_dict: 
                if current_item_name: 
                    target_dict_for_category[current_item_name] = item_details_dict
                elif current_category_internal == overview_category_internal_key: 
                    target_dict_for_category.update(item_details_dict)
                else:
                    logger.warning(f"HParser: Finalizing details but no current_item_name and not overview category ('{current_category_internal}'). Details: {item_details_dict}")
        
        current_item_name = None 
        current_item_detail_lines = [] 

    for line_num, line_text_original_case in enumerate(lines):
        line_stripped = line_text_original_case.strip()

        # ++++++++++++++ CRITICAL DEBUGGING - ADD THIS BLOCK ++++++++++++++
        if "**Overview:**" in line_stripped or "**Locations:**" in line_stripped or \
           "**Factions:**" in line_stripped or "**Systems:**" in line_stripped or \
           "**Lore:**" in line_stripped: # Focus on lines we expect to match

            logger.critical(f"HParser CRITICAL DBG [{line_num+1}] For line_stripped='{repr(line_stripped)}':")
            try:
                # Log UTF-8 bytes
                bytes_utf8 = line_stripped.encode('utf-8', 'replace')
                logger.critical(f"HParser CRITICAL DBG [{line_num+1}]   Bytes (UTF-8): {bytes_utf8}")

                # Log ordinals
                ordinals = [ord(c) for c in line_stripped]
                logger.critical(f"HParser CRITICAL DBG [{line_num+1}]   Ordinals: {ordinals}")

                # Direct comparison to a known-good string
                standard_overview = "**Overview:**"
                if line_stripped == standard_overview:
                    logger.critical(f"HParser CRITICAL DBG [{line_num+1}]   Direct == comparison with '{standard_overview}' PASSED.")
                else:
                    logger.critical(f"HParser CRITICAL DBG [{line_num+1}]   Direct == comparison with '{standard_overview}' FAILED.")
                    logger.critical(f"HParser CRITICAL DBG [{line_num+1}]     Standard '{standard_overview}' ordinals: {[ord(c) for c in standard_overview]}")

                standard_locations = "**Locations:**"
                if line_stripped == standard_locations:
                    logger.critical(f"HParser CRITICAL DBG [{line_num+1}]   Direct == comparison with '{standard_locations}' PASSED.")
                elif "**Locations:**" in line_stripped: # Check if it's this one failing
                    logger.critical(f"HParser CRITICAL DBG [{line_num+1}]   Direct == comparison with '{standard_locations}' FAILED.")
                    logger.critical(f"HParser CRITICAL DBG [{line_num+1}]     Standard '{standard_locations}' ordinals: {[ord(c) for c in standard_locations]}")

            except Exception as e_debug_deep:
                logger.error(f"HParser CRITICAL DBG [{line_num+1}] Error during deep debug logging: {e_debug_deep}")
        # ++++++++++++++ END OF CRITICAL DEBUGGING BLOCK ++++++++++++++

        # Use the category_pattern passed to the function for the main logic
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
            continue

        is_overview_cat = (current_category_internal == overview_category_internal_key)
        line_processed_as_item_header = False

        if not is_overview_cat: 
            potential_item_name = None
            content_as_detail_from_item_line = None 

            item_match_wc = item_pattern_with_content.match(line_stripped)
            if item_match_wc:
                potential_item_name = item_match_wc.group(1).strip()
                content_as_detail_from_item_line = item_match_wc.group(2).strip() 
            else:
                item_match_no = item_pattern_name_only.match(line_stripped)
                if item_match_no:
                    potential_item_name = item_match_no.group(1).strip()
            
            if potential_item_name:
                normalized_potential_item_as_detail_key = potential_item_name.lower().replace(" ", "_")
                is_actually_a_detail_key = False
                if detail_key_map.get(normalized_potential_item_as_detail_key) or \
                   any(isinstance(k_map, re.Pattern) and k_map.match(potential_item_name) for k_map in detail_key_map.keys()):
                    is_actually_a_detail_key = True
                
                if not is_actually_a_detail_key:
                    _finalize_current_item_or_overview_details() 
                    current_item_name = potential_item_name
                    current_item_detail_lines = [] 

                    if content_as_detail_from_item_line: 
                        current_item_detail_lines.append(line_text_original_case) 
                    line_processed_as_item_header = True
        
        if line_processed_as_item_header:
            continue 

        if current_category_internal : 
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

def _parse_triple_list_format(line: str) -> Optional[Tuple[str, str, str]]:
    match = KG_LIST_FORMAT_PATTERN.match(line)
    return match.groups() if match else None

def _parse_triple_kv_format(line: str) -> Optional[Tuple[str, str, str]]:
    match = KG_KV_FORMAT_PATTERN.match(line)
    return match.groups() if match else None

def _parse_triple_pipe_format(line: str) -> Optional[Tuple[str, str, str]]:
    match = KG_PIPE_FORMAT_PATTERN.match(line)
    return match.groups() if match else None

def _parse_triple_comma_format(line: str) -> Optional[Tuple[str, str, str]]:
    if line.startswith("[") and line.endswith("]"): return None
    if line.lower().startswith("subject:") : return None
    parts = [part.strip() for part in line.split(',')]
    if len(parts) == 3 and all(parts):
        if any(p.count(' ') > 5 for p in parts): # Heuristic for too much text
            if not (parts[0].istitle() or parts[0].isupper() or any(c.isdigit() for c in parts[0]) or len(parts[0].split()) <= 3):
                return None
        return tuple(parts) # type: ignore
    return None

KG_TRIPLE_PARSING_STRATEGIES: List[Callable[[str], Optional[Tuple[str, str, str]]]] = [
    _parse_triple_list_format,
    _parse_triple_kv_format,
    _parse_triple_pipe_format,
    _parse_triple_comma_format,
]

def parse_kg_triples_from_text(text_block: str) -> List[List[str]]:
    triples: List[List[str]] = []
    for line_content in text_block.splitlines():
        line = line_content.strip()
        if not line: continue
        parsed_spo: Optional[Tuple[str, str, str]] = None
        for strategy in KG_TRIPLE_PARSING_STRATEGIES:
            result = strategy(line)
            if result:
                parsed_spo = result
                break
        if parsed_spo:
            s, p, o = parsed_spo
            s_cleaned = re.sub(r"^['\"\[\(]+|['\"\]\)]+$", "", s.strip()).strip()
            p_cleaned = re.sub(r"^['\"\[\(]+|['\"\]\)]+$", "", p.strip()).strip()
            o_cleaned = re.sub(r"^['\"\[\(]+|['\"\]\)]+$", "", o.strip()).strip()
            if s_cleaned and p_cleaned and o_cleaned:
                triples.append([s_cleaned, p_cleaned, o_cleaned])
            else:
                logger.warning(f"KG triple parsing: Skipped triple due to empty component after cleaning from line: '{line}' -> S:'{s_cleaned}', P:'{p_cleaned}', O:'{o_cleaned}'")
        elif line and not line.lower().startswith("###"):
            logger.debug(f"KG triple parsing: Could not parse line into a triple: '{line}'")
    if not triples and text_block.strip() and not text_block.lower().startswith(("no kg triples", "none")):
        logger.warning(f"KG triple parsing: No triples extracted from non-empty text block: '{text_block[:200].replace(chr(10), ' ')}...'")
    return triples