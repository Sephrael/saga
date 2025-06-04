# utils/parsing_utils.py
import re
import logging
from typing import List, Dict, Any, Optional, Union, Pattern, Callable, Tuple
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS

logger = logging.getLogger(__name__)

class ParseError(Exception):
    """Custom exception for parsing errors."""
    pass

# Default regex for splitting blocks (e.g., by "---")
DEFAULT_BLOCK_SEPARATOR_REGEX = r'\n\s*---\s*\n'

def split_text_into_blocks(
    text: str,
    separator_regex_str: str = DEFAULT_BLOCK_SEPARATOR_REGEX,
    flags: int = re.MULTILINE
) -> List[str]:
    if not text or not text.strip():
        return []
    blocks = re.split(separator_regex_str, text.strip(), flags=flags)
    return [block.strip() for block in blocks if block and block.strip()]

# --- New RDF Triple Parsing using rdflib ---

def _get_entity_type_and_name_from_uri(uri_ref: URIRef, base_uri: str) -> Dict[str, Optional[str]]:
    logger_func = logging.getLogger(__name__) # Avoid conflict with module-level logger
    uri_str = str(uri_ref)
    name = None

    if uri_str.startswith(base_uri):
        name_part = uri_str[len(base_uri):]
        name = name_part.split('/')[-1].replace('_', ' ')
    else:
        name = uri_str.split('/')[-1].split('#')[-1].replace('_', ' ')

    type_str = None
    if not base_uri.endswith('/'): base_uri_slash = base_uri + '/'
    else: base_uri_slash = base_uri

    if uri_str.startswith(f'{base_uri_slash}Character/'):
        type_str = 'Character'
    elif uri_str.startswith(f'{base_uri_slash}Location/'):
        type_str = 'Location'
    elif uri_str.startswith(f'{base_uri_slash}WorldElement/'):
        type_str = 'WorldElement'
    elif uri_str.startswith(f'{base_uri_slash}Item/'):
        type_str = 'Item'
    elif uri_str.startswith(f'{base_uri_slash}Faction/'):
        type_str = 'Faction'
    elif uri_str.startswith(f'{base_uri_slash}Concept/'):
        type_str = 'Concept'

    if type_str is None and name and name[0].isupper() and f'{base_uri_slash}{name.replace(" ", "_")}' == uri_str :
            logger_func.debug(f"URI {uri_str} resulted in name '{name}' which could be a type itself.")

    return {"type": type_str, "name": name}

def parse_rdf_triples_with_rdflib(text_block: str, rdf_format: str = "turtle", base_uri: str = "http://example.org/saga/") -> List[Dict[str, Any]]:
    logger_func = logging.getLogger(__name__) # Avoid conflict with module-level logger
    triples_list: List[Dict[str, Any]] = []
    if not text_block.strip():
        return triples_list

    if not base_uri.endswith('/'):
        context_base_uri = base_uri + '/'
    else:
        context_base_uri = base_uri

    g = Graph()
    try:
        g.parse(data=text_block, format=rdf_format, publicID=context_base_uri)
    except Exception as e:
        logger_func.error(f"Failed to parse RDF text with rdflib (format: {rdf_format}): {e}", exc_info=True)
        logger_func.error(f"Problematic RDF text block was:\n{text_block[:500]}...")
        return triples_list

    for s, p, o in g:
        predicate_name_parts = str(p).split('/')[-1].split('#')[-1]
        predicate_str = predicate_name_parts.replace('_', ' ')

        s_details = _get_entity_type_and_name_from_uri(s, context_base_uri) if isinstance(s, URIRef) else {"type": "BNode" if isinstance(s, BNode) else "Literal", "name": str(s)}

        # Attempt to get type from rdf:type triple for subject
        for _, _, s_rdf_type_obj in g.triples((s, RDF.type, None)):
            if isinstance(s_rdf_type_obj, URIRef):
                s_details["type"] = str(s_rdf_type_obj).split('/')[-1].split('#')[-1].replace('_', ' ')
                break

        object_entity_payload: Optional[Dict[str, Optional[str]]] = None
        object_literal_payload: Optional[str] = None
        is_literal_object = False

        if isinstance(o, Literal):
            is_literal_object = True
            object_literal_payload = str(o)
        elif isinstance(o, URIRef):
            object_entity_payload = _get_entity_type_and_name_from_uri(o, context_base_uri)
            # Attempt to get type from rdf:type triple for object
            for _, _, o_rdf_type_obj in g.triples((o, RDF.type, None)):
                if isinstance(o_rdf_type_obj, URIRef):
                    # Ensure object_entity_payload is not None before assigning to its key
                    if object_entity_payload is None: object_entity_payload = {} # Should not happen if o is URIRef and _get_entity... works
                    object_entity_payload["type"] = str(o_rdf_type_obj).split('/')[-1].split('#')[-1].replace('_', ' ')
                    break
        elif isinstance(o, BNode):
            object_entity_payload = {"type": "BNode", "name": str(o)}
        else:
            logger_func.warning(f"Unexpected object type: {type(o)} for object {o}")
            continue

        if not s_details.get("name") or not predicate_str:
            logger_func.warning(f"Skipping triple due to missing subject name or predicate: S={s_details}, P={predicate_str}, O_lit={object_literal_payload}, O_ent={object_entity_payload}")
            continue
        if not is_literal_object and (not object_entity_payload or not object_entity_payload.get("name")):
            logger_func.warning(f"Skipping triple due to missing object entity name: S={s_details}, P={predicate_str}, O_ent={object_entity_payload}")
            continue

        triples_list.append({
            "subject": s_details,
            "predicate": predicate_str,
            "object_entity": object_entity_payload,
            "object_literal": object_literal_payload,
            "is_literal_object": is_literal_object
        })
    return triples_list