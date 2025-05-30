# utils/markdown_story_parser.py
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

import config # For FILL_IN_PLACEHOLDER

logger = logging.getLogger(__name__)

# FILL_IN_PLACEHOLDER = config.MARKDOWN_FILL_IN_PLACEHOLDER # Defined in config.py

def _normalize_key(key_str: str) -> str:
    """Normalizes a key string (lowercase, spaces to underscores)."""
    return key_str.strip().lower().replace(" ", "_")

def _get_indentation(line: str) -> int:
    """Returns the number of leading spaces."""
    return len(line) - len(line.lstrip(' '))

def _parse_block_content_recursive(lines_block: List[str], base_indent_for_block: int, debug_path: str = "") -> Any:
    """
    Core recursive parsing function for a block of lines.
    Determines if the block is a list, a dictionary of sub-keys, or a string.
    `base_indent_for_block` is the indent of the construct (key or header) that "owns" this block.
    Sub-keys or list items within this block are often indented relative to the start of the block lines themselves.
    """
    logger.debug(f"{debug_path}_parse_block_content_recursive (base_indent_for_block={base_indent_for_block}) for block:\n{'-'*10}\n" + "\n".join(lines_block) + f"\n{'-'*10}")

    if not lines_block:
        logger.debug(f"{debug_path}Block is empty, returning empty string.")
        return ""

    start_idx, end_idx = 0, len(lines_block) - 1
    while start_idx <= end_idx and not lines_block[start_idx].strip(): start_idx += 1
    while end_idx >= start_idx and not lines_block[end_idx].strip(): end_idx -= 1
    relevant_lines = lines_block[start_idx : end_idx + 1]

    if not relevant_lines:
        logger.debug(f"{debug_path}Block empty after stripping, returning empty string or [Fill-in] if that's all.")
        # If the original block was just "[Fill-in]", preserve it.
        original_block_stripped_content = "\n".join(lines_block).strip()
        if original_block_stripped_content == config.MARKDOWN_FILL_IN_PLACEHOLDER:
            return config.MARKDOWN_FILL_IN_PLACEHOLDER
        return original_block_stripped_content


    # --- Attempt 1: Parse as a LIST ---
    # List items must start with "- " or "* " and maintain or increase indentation.
    is_list_candidate = True
    list_items_parsed = []
    first_item_indent_in_block = -1 # Indent of the first list item relative to the start of lines in this block
    
    # Check if ALL relevant lines are list items
    for line_idx, line_content in enumerate(relevant_lines):
        stripped_line = line_content.strip()
        if not stripped_line: continue # Ignore blank lines within the list structure for this check

        is_list_item_syntax = stripped_line.startswith("- ") or stripped_line.startswith("* ")
        if not is_list_item_syntax:
            is_list_candidate = False
            break
        
        current_item_actual_indent = _get_indentation(line_content)
        if first_item_indent_in_block == -1:
            first_item_indent_in_block = current_item_actual_indent
        
        # All list items in *this immediate list* should have roughly the same indent.
        # Deeper indents would signify sub-lists or multi-line content for a list item,
        # which requires more complex parsing than just collecting `stripped_line[2:]`.
        # For now, we assume simple list items.
        if current_item_actual_indent != first_item_indent_in_block:
            # This could be a multi-line item or sub-list. For simple list parsing, treat as non-list for now.
            # More advanced: collect lines for this item and recurse.
            # For now, this makes it not a candidate for *this specific simple list parsing logic*.
            logger.debug(f"{debug_path}  List item indent changed from {first_item_indent_in_block} to {current_item_actual_indent}. Not treating as simple list. Line: '{line_content}'")
            is_list_candidate = False
            break
        list_items_parsed.append(stripped_line[2:].strip())
            
    if is_list_candidate and list_items_parsed:
        logger.debug(f"{debug_path}  -> Parsed as LIST: {list_items_parsed}")
        # If a list item is just "[Fill-in]", keep it as such.
        return [item if item != config.MARKDOWN_FILL_IN_PLACEHOLDER else config.MARKDOWN_FILL_IN_PLACEHOLDER for item in list_items_parsed]


    # --- Attempt 2: Parse as a DICTIONARY of sub-keys ---
    sub_dictionary: Dict[str, Any] = {}
    active_sub_key_raw: Optional[str] = None
    active_sub_key_line_indent: int = -1 # Actual indent of the sub-key line itself
    current_sub_value_lines: List[str] = []
    
    # The first non-empty line must look like a **Key**: pattern to be a sub-dictionary.
    first_content_line_for_dict_check = relevant_lines[0]
    first_line_key_match = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:(.*)", first_content_line_for_dict_check)

    if first_line_key_match:
        logger.debug(f"{debug_path}  Attempting to parse as SUB-DICTIONARY. First candidate line: '{first_content_line_for_dict_check.strip()}'")
        is_parsing_sub_dict_structure = True
        
        def finalize_sub_dict_item_recursive():
            nonlocal active_sub_key_raw, current_sub_value_lines, sub_dictionary, active_sub_key_line_indent
            if active_sub_key_raw and (current_sub_value_lines or _normalize_key(active_sub_key_raw)): # Ensure key is not empty
                sub_val_debug_path = debug_path + f"  {_normalize_key(active_sub_key_raw)} (sub-key)/"
                logger.debug(f"{debug_path}    Finalizing sub-key '{active_sub_key_raw}'. Recursively parsing its value lines with base_indent={active_sub_key_line_indent}.")
                
                # The base_indent for the sub-key's value is the indent of the sub-key line itself.
                # Content belonging to this sub-key will be indented relative to it.
                sub_value = _parse_block_content_recursive(current_sub_value_lines, active_sub_key_line_indent, sub_val_debug_path)
                sub_dictionary[_normalize_key(active_sub_key_raw)] = sub_value
            
            active_sub_key_raw = None
            current_sub_value_lines = []
            active_sub_key_line_indent = -1 # Reset indent for next potential sub-key
        
        for line_content in relevant_lines:
            current_line_actual_indent = _get_indentation(line_content)
            stripped_line = line_content.strip()
            
            potential_new_sub_key_match = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:(.*)", line_content)

            if potential_new_sub_key_match:
                new_sub_key_line_indent = len(potential_new_sub_key_match.group(1))

                if active_sub_key_raw is None: # First sub-key in this block
                    finalize_sub_dict_item_recursive() # Should be a no-op
                    active_sub_key_raw = potential_new_sub_key_match.group(2).strip()
                    active_sub_key_line_indent = new_sub_key_line_indent
                    current_sub_value_lines = [potential_new_sub_key_match.group(3)] # Start with rest of the line
                    logger.debug(f"{debug_path}    Found first sub-key '{active_sub_key_raw}' at indent {active_sub_key_line_indent}.")
                
                # A new sub-key at the same indentation level as the *first* sub-key found in this block.
                elif new_sub_key_line_indent == active_sub_key_line_indent:
                    finalize_sub_dict_item_recursive()
                    active_sub_key_raw = potential_new_sub_key_match.group(2).strip()
                    # active_sub_key_line_indent remains the same (it's a sibling)
                    current_sub_value_lines = [potential_new_sub_key_match.group(3)]
                    logger.debug(f"{debug_path}    Found sibling sub-key '{active_sub_key_raw}' at indent {active_sub_key_line_indent}.")

                # A new sub-key that is MORE indented than the current active sub-key
                # This means it's a sub-key of the current sub-key's *value*. Add this line to current_sub_value_lines.
                elif new_sub_key_line_indent > active_sub_key_line_indent and active_sub_key_raw is not None:
                    current_sub_value_lines.append(line_content)
                
                # Indentation is less, or active_sub_key_line_indent was not set (should not happen if first_line_key_match was true)
                # This means the sub-dictionary structure is broken or ending.
                else:
                    finalize_sub_dict_item_recursive()
                    is_parsing_sub_dict_structure = False; break 
            
            elif active_sub_key_raw: # This line is a continuation for the active_sub_key's value block
                if not stripped_line: # Preserve blank lines within a multi-line value
                    current_sub_value_lines.append("")
                # Continuation lines for a sub-key's value must be more indented than the sub-key line itself.
                elif current_line_actual_indent > active_sub_key_line_indent:
                    current_sub_value_lines.append(line_content)
                else: # Indentation is not greater, so it's not part of current sub-key's value
                    finalize_sub_dict_item_recursive()
                    is_parsing_sub_dict_structure = False; break
            
            elif stripped_line: # A content line that is not a key and no sub-key is active.
                is_parsing_sub_dict_structure = False; break
        
        finalize_sub_dict_item_recursive() # Finalize any last sub-key
        
        if is_parsing_sub_dict_structure and sub_dictionary:
            logger.debug(f"{debug_path}  -> Parsed as SUB-DICTIONARY: {sub_dictionary}")
            return sub_dictionary
        elif first_line_key_match and not sub_dictionary:
             logger.debug(f"{debug_path}  Was candidate for sub-dict (first line matched key pattern), but no sub-keys were actually parsed. Potential single key-value or fallback to string.")


    # --- Attempt 3: Fallback to multi-line STRING ---
    # This handles single `**Key**: Value` where Value is a simple string, or just plain text.
    # If the original block was just "[Fill-in]", return that.
    block_as_single_string = "\n".join(relevant_lines).strip()
    if block_as_single_string == config.MARKDOWN_FILL_IN_PLACEHOLDER:
        logger.debug(f"{debug_path}  -> Fallback to STRING (is [Fill-in])")
        return config.MARKDOWN_FILL_IN_PLACEHOLDER

    # De-indent the block to the level of its shallowest non-empty line.
    min_content_indent_in_block = float('inf')
    has_any_actual_content = False
    for line_c in relevant_lines:
        if line_c.strip():
            has_any_actual_content = True
            min_content_indent_in_block = min(min_content_indent_in_block, _get_indentation(line_c))
    
    if not has_any_actual_content: # All lines were whitespace
        logger.debug(f"{debug_path}  -> Fallback to STRING (empty after processing relevant lines)")
        return ""
    if min_content_indent_in_block == float('inf'): min_content_indent_in_block = 0 # Should not happen if has_any_actual_content

    processed_string_lines = []
    for line_c in relevant_lines:
        # Remove the common indent from content lines.
        # Preserve blank lines that were part of the original block structure.
        if line_c.strip(): 
            processed_string_lines.append(line_c[min_content_indent_in_block:])
        elif processed_string_lines: # Only keep internal blank lines if content has started
            processed_string_lines.append("") 
            
    final_string = "\n".join(processed_string_lines).strip()
    logger.debug(f"{debug_path}  -> Fallback to STRING: '{final_string[:100].replace(chr(10), ' ')}...'")
    return final_string


def parse_markdown_to_dict(markdown_text: str) -> Dict[str, Any]:
    parsed_data: Dict[str, Any] = {}
    
    text_no_comments = re.sub(r"<!--(.*?)-->", "", markdown_text, flags=re.DOTALL)
    lines = text_no_comments.splitlines()

    active_section_hierarchy: List[Tuple[str, int]] = [] # (normalized_header_name, header_level)
    
    i = 0
    while i < len(lines):
        line_content = lines[i]
        line_actual_indent = _get_indentation(line_content)
        stripped_line = line_content.strip()

        if not stripped_line: 
            i += 1; continue

        header_match = re.match(r"^(#+)\s+(.*)", stripped_line)
        # Key match requires the **Key**: format
        key_match = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:(.*)", line_content)

        if header_match:
            level = len(header_match.group(1))
            header_name_raw = header_match.group(2).strip()
            header_name_norm = _normalize_key(header_name_raw)
            
            # Adjust active_section_hierarchy
            while active_section_hierarchy and active_section_hierarchy[-1][1] >= level:
                active_section_hierarchy.pop()
            active_section_hierarchy.append((header_name_norm, level))
            
            # Navigate to current target_dict
            target_dict = parsed_data
            for name, _ in active_section_hierarchy[:-1]: # Path to parent
                target_dict = target_dict.setdefault(name, {})
            # Ensure the current header exists as a key (will hold its content or sub-keys)
            target_dict.setdefault(header_name_norm, {})
            
            logger.debug(f"Switched to header section: {'/'.join(h[0] for h in active_section_hierarchy)} (level {level}, indent {line_actual_indent})")
            
            # Collect content lines for this header's direct value block
            # Content stops at EOF, new header of same/lesser level, or a non-indented key that's not part of this header block
            header_value_lines: List[str] = []
            line_idx_after_header = i + 1
            while line_idx_after_header < len(lines):
                peek_line = lines[line_idx_after_header]
                peek_stripped = peek_line.strip()
                peek_line_actual_indent = _get_indentation(peek_line)
                
                next_header_match = re.match(r"^(#+)", peek_stripped)
                # Stop if new header of same/lesser level than current one
                if next_header_match and len(next_header_match.group(1)) <= level:
                    break
                
                # Stop if new **Key**: at an indent less than or equal to current header's line indent (means it's a sibling or parent key)
                potential_new_top_key_match = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:", peek_line)
                if potential_new_top_key_match and _get_indentation(peek_line) <= line_actual_indent:
                    break
                
                header_value_lines.append(peek_line)
                line_idx_after_header +=1
            
            if header_value_lines:
                debug_path_val = "/".join(h[0] for h in active_section_hierarchy)
                # The base_indent for a header's content is the header's own line indent
                parsed_val_for_header = _parse_block_content_recursive(header_value_lines, line_actual_indent, debug_path_val + "/")
                
                # If parsed_val_for_header is a dict, merge it into the header's dict.
                # Otherwise, assign it as the value for the header.
                current_header_dict_ref = target_dict[header_name_norm]
                if isinstance(parsed_val_for_header, dict) and isinstance(current_header_dict_ref, dict):
                     current_header_dict_ref.update(parsed_val_for_header)
                elif not current_header_dict_ref: # Header was empty, assign directly
                     target_dict[header_name_norm] = parsed_val_for_header
                # If current_header_dict_ref has content AND parsed_val_for_header is not a dict, this implies mixed content.
                # For simplicity, we might log a warning or overwrite. Here, we assume simple structure or parsed_val is a dict.
                elif parsed_val_for_header: # If header has content from previous keys under it AND this is new non-dict content
                    logger.warning(f"Header '{header_name_norm}' already has content. New non-dictionary content found. Assigning new content. Previous: {current_header_dict_ref}, New: {parsed_val_for_header}")
                    target_dict[header_name_norm] = parsed_val_for_header


            i = line_idx_after_header 
            continue

        elif key_match:
            key_line_indent = len(key_match.group(1)) # Indent of the **Key**: line itself
            key_name_raw = key_match.group(2).strip()
            value_on_same_line_raw = key_match.group(3) # Can be empty

            # Determine current dictionary to add to based on hierarchy
            target_dict = parsed_data
            for name, lvl in active_section_hierarchy:
                target_dict = target_dict.setdefault(name, {})

            value_block_lines_for_key: List[str] = [value_on_same_line_raw] # Start with content on key's line
            line_idx_after_key = i + 1
            while line_idx_after_key < len(lines):
                peek_line = lines[line_idx_after_key]
                peek_line_actual_indent = _get_indentation(peek_line)
                peek_stripped = peek_line.strip()

                # Stop conditions for this key's value block:
                # 1. New header
                if peek_stripped.startswith("#"): break
                # 2. New **Key**: at an indent less than or equal to this key's indent
                next_key_match = re.match(r"^(\s*)\*\*(.*?)\*\*\s*:", peek_line)
                if next_key_match and _get_indentation(peek_line) <= key_line_indent:
                    break
                
                value_block_lines_for_key.append(peek_line)
                line_idx_after_key +=1
            
            key_path_str = "/".join([h[0] for h in active_section_hierarchy] + [_normalize_key(key_name_raw)])
            logger.debug(f"Key '{key_name_raw}' (indent {key_line_indent}) in section {'/'.join(h[0] for h in active_section_hierarchy)}. Parsing its value block with base_indent={key_line_indent}.")
            
            parsed_val_for_key = _parse_block_content_recursive(value_block_lines_for_key, key_line_indent, key_path_str + "/")
            target_dict[_normalize_key(key_name_raw)] = parsed_val_for_key
            
            i = line_idx_after_key 
            continue

        # If line is not a header, not a key, and not empty, it might be unattached content.
        logger.debug(f"Markdown parser: Ignoring unattached line: '{stripped_line}' at path {'/'.join(h[0] for h in active_section_hierarchy)}")
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
        # Return empty dict on error to allow SAGA to proceed with defaults/LLM gen
        return {} 

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
    # Test with user_story_elements.md content
    test_md_content = """<!-- SAGA Story Elements File -->
<!-- Fill in your preferences below. -->
<!-- Use "[Fill-in]" (exactly like that, case-sensitive) for any value you want SAGA to generate. -->
<!-- SAGA will use your concrete entries and generate anything marked "[Fill-in]" or left blank. -->

# Novel Concept
**Title**: Chronoscape Drifters
**Genre**: Time-Travel Adventure with Noir elements
**Theme**: The malleability of history and personal identity
**Logline**: [Fill-in]

# Protagonist
**Name**: Jax Xenobia
**Description**: A jaded ex-temporal agent, haunted by a past mission where a timeline he was supposed to protect fractured catastrophically. He now runs a dusty antique shop in a forgotten eddy of spacetime, trying to escape his memories.
**Character Arc**: From a guilt-ridden recluse to reluctantly accepting his unique ability to perceive 'temporal echoes' of altered events, ultimately confronting the source of the original fracture.
**Traits**:
- Perceptive (especially to temporal anomalies)
- Witty (sarcastic)
- Resourceful
- [Fill-in]
**Initial Status**: Hiding from his past, owner of 'The Hourglass Curios'.
**Relationships**:
  **Lila**: Former partner, presumed lost in the timeline fracture.
  **The Fixer**: Mysterious contact who pulls Jax back into temporal affairs.

# Antagonist
**Name**: Cohort Zero (collective consciousness)
**Description**: An entity from a potential future that believes stabilizing all timelines into one "optimal" path is necessary, even if it means erasing countless others. Appears as shifting, glitch-like figures.
**Motivations**: To prevent a prophesied "Grand Collapse" by imposing absolute order across all of history.

# Conflict
**Summary**: Jax is drawn back into the high-stakes game of temporal manipulation when echoes of his failed mission begin to resurface, threatening his hideaway reality. He must team up with new (and old) allies to stop Cohort Zero from "sanitizing" history, all while battling his own demons and the temptation to alter his personal past.
**Inciting Incident**: An antique device in Jax's shop activates, showing a vivid temporal echo of Lila moments before her disappearance, an echo that shouldn't exist.
**Climax Event Preview**: [Fill-in]

# Plot Points
<!-- Provide as many plot points as you like. SAGA aims for about 12 total. -->
- Jax experiences the anomalous echo and is contacted by The Fixer.
- He investigates the echo, leading him to a clandestine meeting of rogue temporal scholars.
- Cohort Zero agents attack, forcing Jax to use his dormant temporal senses to escape.
- Jax learns Cohort Zero is actively "pruning" minor timelines.
- [Fill-in]
- [Fill-in]
- [Fill-in]
- [Fill-in]
- [Fill-in]
- [Fill-in]
- [Fill-in]
- [Fill-in]


# Setting
**Primary Setting Description**: A multiverse of intersecting timelines, policed (loosely) by various temporal agencies. Key locations include 'Temporal Hubs' (stable nexus points), 'Fracture Zones' (chaotic, unstable realities), and 'Eddy Realities' (isolated, slow-time pockets like Jax's shop).
**Key Locations**:
  **The Hourglass Curios**:
    **Description**: Jax's antique shop, located in an Eddy Reality. Appears normal but contains artifacts from countless timelines.
    **Atmosphere**: Dusty, quiet, filled with the scent of old paper and ozone. A sanctuary that is not as safe as Jax believes.
  **Temporal Hub Prime**:
    **Description**: A vast, crystalline city existing outside normal spacetime, headquarters of the (now largely defunct) Temporal Concord.
    **Atmosphere**: Once bustling and sterile, now largely abandoned and echoing with past glories.
  **The Shifting Sands (Fracture Zone)**:
    **Description**: [Fill-in]
    **Atmosphere**: [Fill-in]

# World Details
**Unique World Feature**: Temporal Echoes
  **Description**: Residual psychic and environmental imprints left by significant timeline alterations or strong emotional events. Jax is uniquely sensitive to them.
  **Rules**:
    - Echoes are fragmented and often misleading.
    - Prolonged exposure can cause disorientation or 'chronal sickness'.
    - Only few individuals can perceive them.
**Key Factions**:
  **The Chronos Wardens (Remnants of Temporal Concord)**:
    **Description**: A scattered group trying to uphold the old laws of non-interference, often clashing with Cohort Zero.
    **Goals**:
      - Preserve timeline integrity (as they see it).
      - Recruit individuals like Jax.
  **Cohort Zero**:
    **Description**: (See Antagonist section)
    **Goals**:
      - Achieve the "Optimal Timeline".
      - Eliminate rogue elements and unpredictable variables (like Jax).
**Relevant Lore**:
  **The Great Timeline Fracture**:
    **Description**: The catastrophic event from Jax's past. Details are hazy and suppressed by temporal authorities.
    **Known Effects**: Caused numerous Eddy Realities to form; source of Jax's trauma.

# Other Key Characters
  **The Fixer**:
    **Description**: An enigmatic information broker who operates across timelines. Gender and appearance seem to shift.
    **Role in Story**: Provides Jax with crucial leads and resources, but their motives are unclear.
    **Traits**:
      - Mysterious
      - Knowledgeable
      - [Fill-in]
    **Relationships**:
      **Jax Xenobia**: Client / Asset
"""
    
    parsed_output = parse_markdown_to_dict(test_md_content)
    import json
    print("\n--- Parsed Output (user_story_elements.md) ---")
    print(json.dumps(parsed_output, indent=2))

    # A more complex nested test
    complex_test_md = """
# Section A
**KeyA1**: Value A1
  This is more of Value A1.
**KeyA2**:
  **SubKeyA2.1**: SubValue A2.1
    This is a sub-value A2.1 line.
  **SubKeyA2.2**:
    - List Item 1 for A2.2
    - List Item 2 for A2.2
**KeyA3**: Another Value A3

## SubSection B Under A
**KeyB1**: Value B1
**KeyB2 (List Form)**:
  - Item X
  - Item Y
    This is part of Item Y. And another line for Y.
  - Item Z
"""
    # print("\n--- Parsed Output (Complex Test) ---")
    # parsed_complex = parse_markdown_to_dict(complex_test_md)
    # print(json.dumps(parsed_complex, indent=2))
    # print("--- End Parsed Output ---")