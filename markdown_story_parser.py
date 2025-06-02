# utils/markdown_story_parser.py
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

import config # For FILL_IN_PLACEHOLDER

logger = logging.getLogger(__name__)

# FILL_IN_PLACEHOLDER = config.MARKDOWN_FILL_IN_PLACEHOLDER # Defined in config.py

def _normalize_key(key_str: str) -> str:
    """Normalizes a key string (lowercase, spaces to underscores)."""
    return key_str.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") # Added parenthesis removal

def _get_indentation(line: str) -> int:
    """Returns the number of leading spaces."""
    return len(line) - len(line.lstrip(' '))

def _parse_block_content_recursive(lines_block: List[str], base_indent_for_block: int, debug_path: str = "") -> Any:
    logger.debug(f"{debug_path}_parse_block_content_recursive (base_indent_for_block={base_indent_for_block}) for block:\n{'-'*10}\n" + "\n".join(lines_block) + f"\n{'-'*10}")

    if not lines_block:
        # logger.debug(f"{debug_path}Block is empty, returning empty string.") # Too noisy
        return ""

    start_idx, end_idx = 0, len(lines_block) - 1
    while start_idx <= end_idx and not lines_block[start_idx].strip(): start_idx += 1
    while end_idx >= start_idx and not lines_block[end_idx].strip(): end_idx -= 1
    relevant_lines = lines_block[start_idx : end_idx + 1]

    if not relevant_lines:
        original_block_stripped_content = "\n".join(lines_block).strip()
        if original_block_stripped_content == config.MARKDOWN_FILL_IN_PLACEHOLDER:
            return config.MARKDOWN_FILL_IN_PLACEHOLDER
        return ""

    # --- Attempt 1: Parse as a LIST ---
    is_list_candidate = True
    list_items_parsed: List[Any] = []
    first_item_marker_indent = -1
    current_item_lines_buffer: List[str] = []

    if not relevant_lines or not (relevant_lines[0].strip().startswith("- ") or relevant_lines[0].strip().startswith("* ")):
        is_list_candidate = False

    if is_list_candidate:
        for line_idx, line_content in enumerate(relevant_lines):
            stripped_line = line_content.strip()
            current_line_actual_indent = _get_indentation(line_content)

            is_current_line_list_item_syntax = stripped_line.startswith("- ") or stripped_line.startswith("* ")

            if is_current_line_list_item_syntax:
                if current_item_lines_buffer:
                    item_content_base_indent = _get_indentation(current_item_lines_buffer[0]) if current_item_lines_buffer and current_item_lines_buffer[0].strip() else first_item_marker_indent + 2
                    item_value = _parse_block_content_recursive(current_item_lines_buffer, item_content_base_indent, debug_path + f"  list_item[{len(list_items_parsed)}]/")
                    list_items_parsed.append(item_value)

                item_content_line = line_content[current_line_actual_indent + 2:]
                current_item_lines_buffer = [item_content_line]
                first_item_marker_indent = current_line_actual_indent

            elif current_item_lines_buffer:
                if current_line_actual_indent > first_item_marker_indent or not stripped_line :
                    current_item_lines_buffer.append(line_content)
                else:
                    is_list_candidate = False
                    if current_item_lines_buffer:
                        item_content_base_indent = _get_indentation(current_item_lines_buffer[0]) if current_item_lines_buffer and current_item_lines_buffer[0].strip() else first_item_marker_indent + 2
                        item_value = _parse_block_content_recursive(current_item_lines_buffer, item_content_base_indent, debug_path + f"  list_item[{len(list_items_parsed)}]/")
                        list_items_parsed.append(item_value)
                        current_item_lines_buffer = []
                    break
            elif stripped_line :
                is_list_candidate = False
                break

        if current_item_lines_buffer and is_list_candidate:
            item_content_base_indent = _get_indentation(current_item_lines_buffer[0]) if current_item_lines_buffer and current_item_lines_buffer[0].strip() else first_item_marker_indent + 2
            item_value = _parse_block_content_recursive(current_item_lines_buffer, item_content_base_indent, debug_path + f"  list_item[{len(list_items_parsed)}]/")
            list_items_parsed.append(item_value)

    if is_list_candidate and list_items_parsed:
        logger.debug(f"{debug_path}  -> Parsed as LIST: {list_items_parsed}")
        return list_items_parsed
    elif is_list_candidate and not list_items_parsed and len(relevant_lines) == 1 and (relevant_lines[0].strip().startswith("- ") or relevant_lines[0].strip().startswith("* ")):
        single_item_content = relevant_lines[0].strip()[2:].strip()
        if not single_item_content or single_item_content == config.MARKDOWN_FILL_IN_PLACEHOLDER:
            logger.debug(f"{debug_path}  -> Parsed as LIST with single [Fill-in] or empty item.")
            return [config.MARKDOWN_FILL_IN_PLACEHOLDER]


    # --- Attempt 2: Parse as a DICTIONARY of sub-keys ---
    sub_dictionary: Dict[str, Any] = {}
    active_sub_key_raw: Optional[str] = None
    active_sub_key_line_indent: int = -1
    current_sub_value_lines: List[str] = []

    first_content_line_for_dict_check = relevant_lines[0]
    first_line_key_match = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:(.*)", first_content_line_for_dict_check)

    if first_line_key_match:
        logger.debug(f"{debug_path}  Attempting to parse as SUB-DICTIONARY. First candidate line: '{first_content_line_for_dict_check.strip()}'")
        is_parsing_sub_dict_structure = True

        indent_of_this_dictionary_level = -1

        def finalize_sub_dict_item_recursive():
            nonlocal active_sub_key_raw, current_sub_value_lines, sub_dictionary, active_sub_key_line_indent
            if active_sub_key_raw:
                sub_val_debug_path = debug_path + f"  {_normalize_key(active_sub_key_raw)} (sub-key)/"
                # The base_indent for a sub-key's value should be relative to that sub-key itself,
                # or more simply, the indent of the first line of its value.
                # Here, active_sub_key_line_indent refers to the indent of the key line.
                # If values are more indented, that's fine.
                value_block_base_indent = _get_indentation(current_sub_value_lines[0]) if current_sub_value_lines and current_sub_value_lines[0].strip() else active_sub_key_line_indent + 2
                sub_value = _parse_block_content_recursive(current_sub_value_lines, value_block_base_indent, sub_val_debug_path)

                sub_dictionary[_normalize_key(active_sub_key_raw)] = sub_value
            active_sub_key_raw = None
            current_sub_value_lines = []
            active_sub_key_line_indent = -1

        for line_content in relevant_lines:
            current_line_actual_indent = _get_indentation(line_content)
            stripped_line = line_content.strip()

            potential_new_sub_key_match = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:(.*)", line_content)

            if potential_new_sub_key_match:
                new_sub_key_line_actual_indent = len(potential_new_sub_key_match.group(1))

                if indent_of_this_dictionary_level == -1:
                    indent_of_this_dictionary_level = new_sub_key_line_actual_indent

                if new_sub_key_line_actual_indent == indent_of_this_dictionary_level:
                    finalize_sub_dict_item_recursive()
                    active_sub_key_raw = potential_new_sub_key_match.group(2).strip()
                    active_sub_key_line_indent = new_sub_key_line_actual_indent
                    current_sub_value_lines = [potential_new_sub_key_match.group(3)]
                    logger.debug(f"{debug_path}    Found sub-key '{active_sub_key_raw}' at line_indent {active_sub_key_line_indent} (level indent for this dict_block: {indent_of_this_dictionary_level}).")
                elif active_sub_key_raw and new_sub_key_line_actual_indent > indent_of_this_dictionary_level :
                    # This line is more indented than the current dict level, so it's part of the current active_sub_key's value.
                    current_sub_value_lines.append(line_content)
                else: # New key at a lesser indent, or no active key and this one doesn't match current level
                    finalize_sub_dict_item_recursive()
                    is_parsing_sub_dict_structure = False; break
            elif active_sub_key_raw:
                if not stripped_line: # Empty line, potentially part of multi-line value
                    current_sub_value_lines.append("")
                elif current_line_actual_indent > active_sub_key_line_indent : # More indented than the key line itself
                    current_sub_value_lines.append(line_content)
                else: # Less or equally indented, new structure or end of block
                    finalize_sub_dict_item_recursive()
                    is_parsing_sub_dict_structure = False; break
            elif stripped_line: # No active sub-key, and this line is not a new sub-key at the current level
                is_parsing_sub_dict_structure = False; break

        finalize_sub_dict_item_recursive()

        if is_parsing_sub_dict_structure and sub_dictionary:
            logger.debug(f"{debug_path}  -> Parsed as SUB-DICTIONARY: {sub_dictionary}")
            return sub_dictionary

    # --- Attempt 3: Fallback to multi-line STRING ---
    block_as_single_string = "\n".join(relevant_lines).strip()
    if block_as_single_string == config.MARKDOWN_FILL_IN_PLACEHOLDER:
        logger.debug(f"{debug_path}  -> Fallback to STRING (is [Fill-in])")
        return config.MARKDOWN_FILL_IN_PLACEHOLDER

    min_content_indent_in_block = float('inf')
    has_any_actual_content = False
    for line_c in relevant_lines:
        if line_c.strip():
            has_any_actual_content = True
            min_content_indent_in_block = min(min_content_indent_in_block, _get_indentation(line_c))

    if not has_any_actual_content:
        # logger.debug(f"{debug_path}  -> Fallback to STRING (empty after processing relevant lines)") # Too noisy
        return ""
    if min_content_indent_in_block == float('inf'): min_content_indent_in_block = 0

    processed_string_lines = []
    for line_c_idx, line_c in enumerate(relevant_lines):
        if line_c.strip():
            processed_string_lines.append(line_c[min_content_indent_in_block:])
        elif processed_string_lines: # Keep empty lines if they are part of a multi-line string
            processed_string_lines.append("")

    final_string = "\n".join(processed_string_lines).strip()
    logger.debug(f"{debug_path}  -> Fallback to STRING: '{final_string[:100].replace(chr(10), ' ')}...'")
    return final_string

# Helper function to navigate/create dictionary path
def _get_or_create_path(root_dict: Dict[str, Any], path_keys: List[str]) -> Dict[str, Any]:
    d = root_dict
    for key in path_keys:
        node = d.get(key)
        if not isinstance(node, dict):
            # This might happen if a header was previously assigned a direct value (e.g., a list)
            # and now a sub-header or key is trying to be placed under it.
            logger.debug(f"_get_or_create_path: Node for key '{key}' is not a dict (type: {type(node)}). Creating/overwriting as dict.")
            node = {}
            d[key] = node
        d = node
    return d

def parse_markdown_to_dict(markdown_text: str) -> Dict[str, Any]:
    parsed_data: Dict[str, Any] = {}
    text_no_comments = re.sub(r"<!--(.*?)-->", "", markdown_text, flags=re.DOTALL)
    lines = text_no_comments.splitlines()
    active_section_hierarchy: List[Tuple[str, int, int]] = [] # (norm_name, level, line_indent_of_header_declaration_line)

    i = 0
    while i < len(lines):
        line_content_original = lines[i]
        current_line_actual_indent = _get_indentation(line_content_original)
        stripped_line = line_content_original.strip()

        if not stripped_line:
            i += 1; continue

        header_match = re.match(r"^(#+)\s+(.*)", stripped_line)
        # **Key**: Value
        key_match = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:(.*)", line_content_original)


        if header_match:
            level = len(header_match.group(1))
            header_name_raw = header_match.group(2).strip()
            header_name_norm = _normalize_key(header_name_raw)

            # Pop headers from hierarchy if new header is of same or higher level (smaller # count)
            while active_section_hierarchy and active_section_hierarchy[-1][1] >= level:
                active_section_hierarchy.pop()

            parent_for_new_header = _get_or_create_path(parsed_data, [h[0] for h in active_section_hierarchy])
            parent_for_new_header.setdefault(header_name_norm, {}) # Initialize as dict
            active_section_hierarchy.append((header_name_norm, level, current_line_actual_indent))

            logger.debug(f"Switched to header section: {'/'.join(h[0] for h in active_section_hierarchy)} (level {level}, header_line_indent {current_line_actual_indent})")
            i += 1
            continue

        elif key_match:
            key_line_actual_indent = len(key_match.group(1)) # Indent of the key itself
            key_name_raw = key_match.group(2).strip()
            key_name_norm = _normalize_key(key_name_raw)
            value_on_same_line_raw = key_match.group(3)

            # Determine the correct parent for this key based on hierarchy and key's indent
            # A key belongs under the deepest header whose own line_indent is <= key's line_indent
            temp_hierarchy_path_keys = []
            for h_name, h_level, h_line_indent in active_section_hierarchy:
                if h_line_indent <= key_line_actual_indent:
                    temp_hierarchy_path_keys.append(h_name)
                else: # This header is more indented than the key, so key is not under it
                    break
            
            target_dict_for_key = _get_or_create_path(parsed_data, temp_hierarchy_path_keys)
            current_key_parent_path_str = "/".join(temp_hierarchy_path_keys) if temp_hierarchy_path_keys else "ROOT"


            # Collect lines for the key's value block
            value_block_lines_for_key: List[str] = [value_on_same_line_raw]
            line_idx_after_key = i + 1
            while line_idx_after_key < len(lines):
                peek_line = lines[line_idx_after_key]
                peek_line_actual_indent = _get_indentation(peek_line)
                peek_stripped = peek_line.strip()

                if peek_stripped.startswith("#"): # New header always stops value block
                    break

                next_key_match_on_peek = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:", peek_line)
                if next_key_match_on_peek:
                    next_key_indent_peek = len(next_key_match_on_peek.group(1))
                    # If the next key is at the same or lesser indent than the *current key's indent*
                    if next_key_indent_peek <= key_line_actual_indent:
                        break
                
                # If it's not a new header or a new sibling/parent key, it's part of the current key's value
                # This includes more indented lines, list items, or blank lines within the value
                if peek_line_actual_indent > key_line_actual_indent or \
                   (peek_stripped.startswith(("- ", "* ")) and peek_line_actual_indent >= key_line_actual_indent) or \
                   not peek_stripped:
                     value_block_lines_for_key.append(peek_line)
                else: # Line is less indented and not fitting value continuation criteria
                    break
                line_idx_after_key +=1

            if value_block_lines_for_key and not value_block_lines_for_key[0].strip() and len(value_block_lines_for_key) > 1:
                 value_block_lines_for_key.pop(0)
            elif value_block_lines_for_key and not value_block_lines_for_key[0].strip() and len(value_block_lines_for_key) == 1:
                 value_block_lines_for_key = []

            key_path_str_for_debug = f"{current_key_parent_path_str}/{key_name_norm}"
            logger.debug(f"Key '{key_name_raw}' (key_line_indent {key_line_actual_indent}) assigned to section '{current_key_parent_path_str}'. Parsing its value block. Base indent for value parsing for _parse_block_content_recursive: {key_line_actual_indent}.")

            parsed_val_for_key = _parse_block_content_recursive(value_block_lines_for_key, key_line_actual_indent, key_path_str_for_debug + "/")
            target_dict_for_key[key_name_norm] = parsed_val_for_key

            i = line_idx_after_key
            continue

        # Logic for handling direct content under a header (e.g., a list of plot points)
        # This should only trigger if the current line is not a key_match and there's an active header
        if active_section_hierarchy and stripped_line:
            current_header_name_norm, current_header_level, current_header_line_indent = active_section_hierarchy[-1]
            
            parent_path_keys_for_header = [h[0] for h in active_section_hierarchy[:-1]]
            parent_dict_for_header_value = _get_or_create_path(parsed_data, parent_path_keys_for_header)
            current_header_node = parent_dict_for_header_value.get(current_header_name_norm)

            # Only assign direct value if the header's node is an empty dict (meaning it hasn't had keys assigned yet)
            if isinstance(current_header_node, dict) and not current_header_node:
                # Check if the current line is intended as direct content for this header
                is_list_item_current = stripped_line.startswith("- ") or stripped_line.startswith("* ")
                
                # Content belongs to this header if it's more indented OR
                # if it's a list item starting at or after header's own line declaration indent
                if current_line_actual_indent > current_header_line_indent or \
                   (is_list_item_current and current_line_actual_indent >= current_header_line_indent):

                    logger.debug(f"Header '{current_header_name_norm}' (h_indent {current_header_line_indent}) seems to have direct content starting with: '{stripped_line}' (line_indent {current_line_actual_indent})")
                    direct_value_lines = [line_content_original]
                    idx_after_direct_val_start = i + 1
                    while idx_after_direct_val_start < len(lines):
                        peek_line = lines[idx_after_direct_val_start]
                        peek_line_actual_indent = _get_indentation(peek_line)
                        peek_stripped = peek_line.strip()

                        # Stop if a new header of same or higher significance (smaller level number) is found at a lesser or equal indent
                        if peek_stripped.startswith("#"):
                            peek_header_level_match = re.match(r"^(#+)", peek_stripped)
                            if peek_header_level_match:
                                peek_header_level = len(peek_header_level_match.group(1))
                                if peek_header_level <= current_header_level and peek_line_actual_indent <= current_header_line_indent:
                                    break
                        
                        # Stop if a new key is found at an indent less than or equal to the current header's indent
                        next_key_match_peek = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:", peek_line)
                        if next_key_match_peek and _get_indentation(peek_line) <= current_header_line_indent:
                            break
                        
                        # Continue collecting lines if they are more indented than the header or are list items at appropriate indent
                        is_list_item_peek = peek_stripped.startswith("- ") or peek_stripped.startswith("* ")
                        if peek_line_actual_indent > current_header_line_indent or \
                           (is_list_item_peek and peek_line_actual_indent >= current_header_line_indent) or \
                           not peek_stripped: # Include empty lines
                            direct_value_lines.append(peek_line)
                        else: # Line is less indented and not part of the direct value
                            break
                        idx_after_direct_val_start += 1
                    
                    # The base_indent for recursive call should be the indent of the header itself
                    parsed_direct_value = _parse_block_content_recursive(direct_value_lines, current_header_line_indent, current_header_name_norm + "/(direct_value)/")
                    parent_dict_for_header_value[current_header_name_norm] = parsed_direct_value
                    i = idx_after_direct_val_start
                    continue

        logger.debug(f"Markdown parser: Ignoring line (no match): '{stripped_line}' at path {'/'.join(h[0] for h in active_section_hierarchy)}")
        i += 1

    return parsed_data


def load_and_parse_markdown_story_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads and parses the Markdown story file."""
    if not filepath.endswith(".md"):
        logger.error(f"File specified is not a Markdown file: {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content.strip():
            logger.warning(f"Markdown story file is empty: {filepath}")
            return {}

        parsed_dict = parse_markdown_to_dict(content)
        logger.info(f"Successfully parsed Markdown story file: {filepath}")
        return parsed_dict
    except FileNotFoundError:
        logger.warning(f"Markdown story file '{filepath}' not found. Will proceed without user preferences if this is optional.")
        return None
    except Exception as e:
        logger.error(f"Error parsing Markdown story file {filepath}: {e}", exc_info=True)
        return {}

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
    test_md_content = """<!-- SAGA Story Elements File -->
# Novel Concept
**Title**: Chronoscape Drifters
**Genre**: Time-Travel Adventure with Noir elements

# Protagonist
**Name**: Jax Xenobia
**Description**: A jaded ex-temporal agent.
  He now runs a dusty antique shop.
**Traits**:
  - Perceptive
  - Witty
  - Resourceful
  - [Fill-in]
**Relationships**:
  **Lila**: Former partner.
    **Status**: Presumed lost.
  **The Fixer**: Mysterious contact.

# Setting
## Primary Setting
**Description**: A multiverse of intersecting timelines.
## Key Locations
### The Hourglass Curios
**Description**: Jax's antique shop.
**Atmosphere**: Dusty, quiet.
### Temporal Hub Prime
**Description**: A vast, crystalline city.
**Atmosphere**: Once bustling, now abandoned.
    **Sub Detail Under Atmosphere**: Still hums with residual energy.
    **Another Sub Detail**:
        - Point 1
        - Point 2
"""

    parsed_output = parse_markdown_to_dict(test_md_content)
    import json
    print("\n--- Parsed Output (user_story_elements.md STYLE TEST) ---")
    print(json.dumps(parsed_output, indent=2))

    llm_like_output_from_log = """## Overview
**Description**: *The Shattered Spire* is a cosmic tragedy where humanity's ambition to
transcend time and space has birthed a colossal, ringed megastructure orbiting a dying
star. This structure, once an awe-inspiring beacon of progress, now stands as a symbol of
hubris and decay. The Spire’s core pulses with unstable energy, threatening to collapse the
entire system, while its outer rings are wild, lawless territories filled with scavengers,
rebels, and remnants of lost civilizations. The surrounding space is a graveyard of failed
experiments and ancient ruins, each whispering tales of those who sought to control time
but were consumed by it.
**Atmosphere**: A blend of eerie silence, humming energy, and the haunting echoes of past
lives. The Spire radiates an unsettling presence, with gravity anomalies and temporal
distortions creating a world where time is not linear but fractured.
**Tone**: Melancholic, mysterious, and tinged with existential dread.

## Locations
### The Inner Sanctum
**Description**: A tightly controlled sector at the heart of the Spire, housing its central
power core and the ruling elite. It is a place of pristine architecture, artificial light,
and strict order, maintained by advanced automation and surveillance. The inner sanctum is
where the Spire's true purpose is hidden from the general populace.
**Atmosphere**: Cold, sterile, and suffocating. A constant hum of energy permeates the air,
and time seems to move at a different pace than in the outer rings.
**Features**:
  - A central monolith known as the *Eternal Core*
  - holographic archives
  - remnants of long-dead machinery.
### The Obsidian Veins
**Description**: A network of dark tunnels that snake through the lower levels of the Veil,
once used for transporting raw materials but now home to rogue factions and unstable
temporal rifts.
**Atmosphere**: Oppressive and claustrophobic, with flickering lights and echoes of ancient
voices.
**Features**:
  - Hidden chambers
  - malfunctioning transport tubes
  - unpredictable time loops that trap travelers indefinitely.
"""
    print("\n--- Parsed Output (LLM-like From Log with list for Features) ---")
    parsed_llm_like = parse_markdown_to_dict(llm_like_output_from_log)
    print(json.dumps(parsed_llm_like, indent=2))


    plot_points_test = """
# Plot Points
- Point 1
- Point 2
  Continuation of Point 2.
  Another line for Point 2.
    - Sub-point A for Point 2
    - Sub-point B for Point 2
- Point 3
"""
    print("\n--- Parsed Output (Plot Points Test) ---")
    parsed_plot_points = parse_markdown_to_dict(plot_points_test)
    print(json.dumps(parsed_plot_points, indent=2))