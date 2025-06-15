# data_access/kg_queries.py
import re
from typing import Any, Dict, List, Optional, Tuple

import structlog

import config
from core_db.base_db_manager import neo4j_manager
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_REL_CHAPTER_ADDED,
)

logger = structlog.get_logger(__name__)


def _normalize_label(label: str) -> str:
    """Return a sanitized label string with first letter capitalized."""
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", label.strip().casefold())
    return cleaned.capitalize() if cleaned else ""


def normalize_relationship_type(rel: str) -> str:
    """Normalize relationship types to uppercase snake case."""
    rel_norm = re.sub(r"[^a-z0-9]+", "_", rel.strip().casefold())
    rel_norm = re.sub(r"_+", "_", rel_norm).strip("_")
    return rel_norm.upper()


def _get_cypher_labels(entity_type: Optional[str]) -> str:
    """Helper to create a Cypher label string (e.g., :Character:Entity or :Person:Character:Entity)."""

    entity_label_suffix = ":Entity"  # All nodes get this
    specific_labels_parts = []

    if entity_type and entity_type.strip():
        original_type_capitalized = entity_type.strip().capitalize()
        sanitized_specific_type = _normalize_label(entity_type)

        if sanitized_specific_type and sanitized_specific_type != "Entity":
            # Add the sanitized specific type label unless it's "Character" (which is handled next)
            if sanitized_specific_type != "Character":
                specific_labels_parts.append(f":{sanitized_specific_type}")

            # Add :Character label if the original type was "Person" OR
            # if the sanitized type is "Character" itself.
            if (
                original_type_capitalized == "Person"
                or sanitized_specific_type == "Character"
            ):
                # Add :Character if not already added (e.g. if original was "Character")
                if ":Character" not in specific_labels_parts:
                    specific_labels_parts.append(":Character")

    # Order: :Character (if present), then other specific labels (e.g., :Person), then :Entity
    # Remove duplicates and establish order
    final_ordered_labels = []
    if ":Character" in specific_labels_parts:
        final_ordered_labels.append(":Character")

    for label in specific_labels_parts:
        if label not in final_ordered_labels:
            final_ordered_labels.append(label)

    if not final_ordered_labels:
        return entity_label_suffix  # Just ":Entity"

    return "".join(final_ordered_labels) + entity_label_suffix


async def add_kg_triples_batch_to_db(
    structured_triples_data: List[Dict[str, Any]],
    chapter_number: int,
    is_from_flawed_draft: bool,
):
    if not structured_triples_data:
        logger.info("Neo4j: add_kg_triples_batch_to_db: No structured triples to add.")
        return

    statements_with_params: List[Tuple[str, Dict[str, Any]]] = []

    for triple_dict in structured_triples_data:
        subject_info = triple_dict.get("subject")
        predicate_str = triple_dict.get("predicate")

        object_entity_info = triple_dict.get("object_entity")
        object_literal_val = triple_dict.get(
            "object_literal"
        )  # This will be a string from parsing
        is_literal_object = triple_dict.get("is_literal_object", False)

        if not (
            subject_info
            and isinstance(subject_info, dict)
            and subject_info.get("name")
            and predicate_str
        ):
            logger.warning(
                f"Neo4j (Batch): Invalid subject or predicate in triple dict: {triple_dict}"
            )
            continue

        subject_name = str(subject_info["name"]).strip()
        subject_type = subject_info.get(
            "type"
        )  # This is a string like "Character", "WorldElement", etc.
        predicate_clean = normalize_relationship_type(str(predicate_str))

        if not all([subject_name, predicate_clean]):
            logger.warning(
                f"Neo4j (Batch): Empty subject name or predicate after stripping: {triple_dict}"
            )
            continue

        subject_labels_cypher = _get_cypher_labels(subject_type)

        # Base parameters for the relationship
        rel_props = {
            "type": predicate_clean,
            KG_REL_CHAPTER_ADDED: chapter_number,
            KG_IS_PROVISIONAL: is_from_flawed_draft,
            "confidence": 1.0,  # Default confidence
            # Add other relationship metadata if available
        }

        params = {"subject_name_param": subject_name, "rel_props_param": rel_props}

        if is_literal_object:
            if object_literal_val is None:
                logger.warning(
                    f"Neo4j (Batch): Literal object is None for triple: {triple_dict}"
                )
                continue

            # For literal objects, merge/create a ValueNode.
            # The ValueNode is unique by its string value and type 'Literal'.
            params["object_literal_value_param"] = str(
                object_literal_val
            )  # Ensure it's a string for ValueNode value
            params["value_node_type_param"] = (
                "Literal"  # Generic type for these literal ValueNodes
            )

            query = f"""
            MERGE (s{subject_labels_cypher} {{name: $subject_name_param}})
            MERGE (o:Entity:ValueNode {{value: $object_literal_value_param, type: $value_node_type_param}})
                ON CREATE SET o.created_ts = timestamp()

            MERGE (s)-[r:DYNAMIC_REL]->(o)
            SET r = $rel_props_param,
                r.created_at = COALESCE(r.created_at, timestamp()),
                r.last_updated = timestamp()
            """
            statements_with_params.append((query, params))

        elif (
            object_entity_info
            and isinstance(object_entity_info, dict)
            and object_entity_info.get("name")
        ):
            object_name = str(object_entity_info["name"]).strip()
            object_type = object_entity_info.get(
                "type"
            )  # String like "Location", "Item"
            if not object_name:
                logger.warning(
                    f"Neo4j (Batch): Empty object name for entity object in triple: {triple_dict}"
                )
                continue

            object_labels_cypher = _get_cypher_labels(object_type)
            params["object_name_param"] = object_name

            query = f"""
            MERGE (s{subject_labels_cypher} {{name: $subject_name_param}})
            MERGE (o{object_labels_cypher} {{name: $object_name_param}}) 
                ON CREATE SET o.created_ts = timestamp() // Set timestamp if object node is newly created
            
            MERGE (s)-[r:DYNAMIC_REL]->(o)
            SET r = $rel_props_param,
                r.created_at = COALESCE(r.created_at, timestamp()),
                r.last_updated = timestamp()
            """
            statements_with_params.append((query, params))
        else:
            logger.warning(
                f"Neo4j (Batch): Invalid or missing object information in triple dict: {triple_dict}"
            )
            continue

    if not statements_with_params:
        logger.info(
            "Neo4j: add_kg_triples_batch_to_db: No valid statements generated from triples."
        )
        return

    try:
        await neo4j_manager.execute_cypher_batch(statements_with_params)
        logger.info(
            f"Neo4j: Batch processed {len(statements_with_params)} KG triple statements."
        )
    except Exception as e:
        # Log first few problematic params for debugging, if any
        first_few_params_str = (
            str([p_tuple[1] for p_tuple in statements_with_params[:2]])
            if statements_with_params
            else "N/A"
        )
        logger.error(
            f"Neo4j: Error in batch adding KG triples. First few params: {first_few_params_str}. Error: {e}",
            exc_info=True,
        )
        raise


async def query_kg_from_db(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    obj_val: Optional[str] = None,
    chapter_limit: Optional[int] = None,
    include_provisional: bool = True,
    limit_results: Optional[int] = None,
) -> List[Dict[str, Any]]:
    conditions = []
    parameters: Dict[str, Any] = {}
    match_clause = "MATCH (s:Entity)-[r:DYNAMIC_REL]->(o) "

    if subject is not None:
        conditions.append("s.name = $subject_param")
        parameters["subject_param"] = subject.strip()
    if predicate is not None:
        conditions.append("r.type = $predicate_param")
        parameters["predicate_param"] = normalize_relationship_type(predicate)
    if obj_val is not None:
        obj_val_stripped = obj_val.strip()
        conditions.append(
            """
            ( (o:ValueNode AND o.value = $object_param ) OR
              (NOT o:ValueNode AND o.name = $object_param)
            )
        """
        )
        parameters["object_param"] = obj_val_stripped
    if chapter_limit is not None:
        conditions.append(f"r.{KG_REL_CHAPTER_ADDED} <= $chapter_limit_param")
        parameters["chapter_limit_param"] = chapter_limit
    if not include_provisional:
        conditions.append(
            f"(r.{KG_IS_PROVISIONAL} = FALSE OR r.{KG_IS_PROVISIONAL} IS NULL)"
        )

    where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

    return_clause = f"""
    RETURN s.name AS subject, 
           r.type AS predicate, 
           CASE WHEN o:ValueNode THEN o.value ELSE o.name END AS object,
           CASE WHEN o:ValueNode THEN 'Literal' ELSE labels(o)[0] END AS object_type, // Primary label or 'Literal'
           r.{KG_REL_CHAPTER_ADDED} AS {KG_REL_CHAPTER_ADDED}, 
           r.confidence AS confidence, 
           r.{KG_IS_PROVISIONAL} AS {KG_IS_PROVISIONAL}
    """
    order_clause = f" ORDER BY r.{KG_REL_CHAPTER_ADDED} DESC, r.confidence DESC"
    limit_clause_str = (
        f" LIMIT {int(limit_results)}"
        if limit_results is not None and limit_results > 0
        else ""
    )

    full_query = (
        match_clause + where_clause + return_clause + order_clause + limit_clause_str
    )
    try:
        results = await neo4j_manager.execute_read_query(full_query, parameters)
        triples_list: List[Dict[str, Any]] = (
            [dict(record) for record in results] if results else []
        )
        logger.debug(
            f"Neo4j: KG query returned {len(triples_list)} results. Query: '{full_query[:200]}...' Params: {parameters}"
        )
        return triples_list
    except Exception as e:
        logger.error(
            f"Neo4j: Error querying KG. Query: '{full_query[:200]}...', Params: {parameters}, Error: {e}",
            exc_info=True,
        )
        return []


async def get_most_recent_value_from_db(
    subject: str,
    predicate: str,
    chapter_limit: Optional[int] = None,
    include_provisional: bool = False,
) -> Optional[Any]:
    if not subject.strip() or not predicate.strip():
        logger.warning(
            f"Neo4j: get_most_recent_value_from_db: empty subject or predicate. S='{subject}', P='{predicate}'"
        )
        return None

    results = await query_kg_from_db(
        subject=subject,
        predicate=predicate,
        chapter_limit=chapter_limit,
        include_provisional=include_provisional,
        limit_results=1,
    )
    if results and results[0] and "object" in results[0]:
        value = results[0]["object"]
        # Attempt to convert to number if it looks like one, as ValueNode.value stores as string from current triple parsing
        if isinstance(value, str):
            if re.match(r"^-?\d+$", value):
                value = int(value)
            elif re.match(r"^-?\d*\.\d+$", value):
                value = float(value)
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False

        logger.debug(
            f"Neo4j: Found most recent value for ('{subject}', '{predicate}'): '{value}' (type: {type(value)}) from Ch {results[0].get(KG_REL_CHAPTER_ADDED, 'N/A')}, Prov: {results[0].get(KG_IS_PROVISIONAL)}"
        )
        return value

    logger.debug(
        f"Neo4j: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}, include_provisional={include_provisional}."
    )
    return None


async def get_novel_info_property_from_db(property_key: str) -> Optional[Any]:
    """Return a property value from the NovelInfo node."""
    if not property_key.strip():
        logger.warning("Neo4j: empty property key for NovelInfo query")
        return None

    novel_id_param = config.MAIN_NOVEL_INFO_NODE_ID
    query = f"MATCH (ni:NovelInfo:Entity {{id: $novel_id_param}}) RETURN ni.{property_key} AS value"
    try:
        results = await neo4j_manager.execute_read_query(
            query, {"novel_id_param": novel_id_param}
        )
        if results and results[0] and "value" in results[0]:
            return results[0]["value"]
    except Exception as e:  # pragma: no cover - narrow DB errors
        logger.error(
            f"Neo4j: Error retrieving NovelInfo property '{property_key}': {e}",
            exc_info=True,
        )
    return None


async def get_chapter_context_for_entity(
    entity_name: Optional[str] = None, entity_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Finds chapters where an entity was mentioned or involved to provide context for enrichment.
    Searches by name for Characters/ValueNodes or by ID for WorldElements.
    """
    if not entity_name and not entity_id:
        return []

    match_clause = (
        "MATCH (e {id: $id_param})" if entity_id else "MATCH (e {name: $name_param})"
    )
    params = {"id_param": entity_id} if entity_id else {"name_param": entity_name}

    # CORRECTED QUERY
    query = f"""
    {match_clause}

    // Get all paths to potential chapter number sources
    OPTIONAL MATCH (e)-[]->(event) WHERE (event:DevelopmentEvent OR event:WorldElaborationEvent) AND event.chapter_updated IS NOT NULL
    OPTIONAL MATCH (e)-[r:DYNAMIC_REL]-() WHERE r.chapter_added IS NOT NULL

    // Collect all numbers into one list, then process
    WITH
      CASE WHEN e.created_chapter IS NOT NULL THEN [e.created_chapter] ELSE [] END as created_chapter_list,
      COLLECT(DISTINCT event.chapter_updated) as event_chapters,
      COLLECT(DISTINCT r.chapter_added) as rel_chapters

    // Combine, filter out nulls, unwind, get distinct
    WITH created_chapter_list + event_chapters + rel_chapters as all_chapters
    UNWIND all_chapters as chapter_num
    WITH DISTINCT chapter_num
    WHERE chapter_num IS NOT NULL AND chapter_num > 0

    // Now fetch the chapter data
    MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL} {{number: chapter_num}})
    RETURN c.number as chapter_number, c.summary as summary, c.text as text
    ORDER BY c.number DESC
    LIMIT 5 // Limit context to most recent 5 chapters
    """
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        return results if results else []
    except Exception as e:
        logger.error(
            f"Error getting chapter context for entity '{entity_name or entity_id}': {e}",
            exc_info=True,
        )
        return []


async def find_contradictory_trait_characters(
    contradictory_trait_pairs: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """
    Finds characters who have contradictory traits based on a provided list of pairs.
    e.g. [('Brave', 'Cowardly'), ('Honest', 'Deceitful')]
    """
    if not contradictory_trait_pairs:
        return []

    all_findings = []
    for trait1, trait2 in contradictory_trait_pairs:
        query = """
        MATCH (c:Character)-[:HAS_TRAIT]->(t1:Trait {name: $trait1_param}),
              (c)-[:HAS_TRAIT]->(t2:Trait {name: $trait2_param})
        RETURN c.name AS character_name, t1.name AS trait1, t2.name AS trait2
        """
        params = {"trait1_param": trait1, "trait2_param": trait2}
        try:
            results = await neo4j_manager.execute_read_query(query, params)
            if results:
                all_findings.extend(results)
        except Exception as e:
            logger.error(
                f"Error checking for contradictory traits '{trait1}' vs '{trait2}': {e}",
                exc_info=True,
            )

    return all_findings


async def find_post_mortem_activity() -> List[Dict[str, Any]]:
    """
    Finds characters who have relationships or activities recorded in chapters
    after they were marked as dead.
    """
    query = """
    MATCH (c:Character)-[death_rel:DYNAMIC_REL {type: 'IS_DEAD'}]->()
    WHERE death_rel.is_provisional = false
    WITH c, death_rel.chapter_added AS death_chapter

    MATCH (c)-[activity_rel:DYNAMIC_REL]->()
    WHERE activity_rel.chapter_added > death_chapter
      AND NOT activity_rel.type IN ['IS_REMEMBERED_AS', 'WAS_FRIEND_OF'] // Exclude retrospective rels
    RETURN DISTINCT c.name as character_name,
           death_chapter,
           collect(
             {
               activity_type: activity_rel.type,
               activity_chapter: activity_rel.chapter_added
             }
           ) AS post_mortem_activities
    LIMIT 20
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return results if results else []
    except Exception as e:
        logger.error(f"Error checking for post-mortem activity: {e}", exc_info=True)
        return []


async def find_candidate_duplicate_entities(
    similarity_threshold: float = 0.85, limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Finds pairs of entities with similar names using APOC's Levenshtein distance.
    This requires the APOC plugin to be installed in Neo4j.
    """
    # This query now uses `apoc.coll.max` which is the correct scalar function
    # for finding the max value within a single row's context.
    query = """
    MATCH (e1:Entity), (e2:Entity)
    WHERE id(e1) < id(e2)
      AND e1.name IS NOT NULL AND e2.name IS NOT NULL
      AND NOT e1:ValueNode AND NOT e2:ValueNode
    WITH e1, e2, apoc.text.distance(e1.name, e2.name) AS distance
    
    // CORRECTED LINE: Use apoc.coll.max on a list of the two sizes.
    WITH e1, e2, distance, apoc.coll.max([size(e1.name), size(e2.name)]) as max_len

    // Calculate similarity using the new max_len variable
    WHERE (1 - (distance / toFloat(max_len))) >= $threshold
    
    RETURN
      e1.id AS id1, e1.name AS name1, labels(e1) AS labels1,
      e2.id AS id2, e2.name AS name2, labels(e2) AS labels2,
      (1 - (distance / toFloat(max_len))) as similarity
    ORDER BY similarity DESC
    LIMIT $limit
    """
    params = {"threshold": similarity_threshold, "limit": limit}
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        return results if results else []
    except Exception as e:
        if "apoc.text.distance" in str(e) or "apoc.coll.max" in str(e):
            logger.error(
                "A required APOC function was not found. "
                "Please ensure the full APOC Extended plugin is installed."
            )
        else:
            logger.error(
                f"Error finding candidate duplicate entities: {e}", exc_info=True
            )
        return []


async def get_entity_context_for_resolution(
    entity_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Gathers comprehensive context for an entity to help an LLM decide on a merge.
    """
    query = """
    MATCH (e:Entity {id: $entity_id})
    OPTIONAL MATCH (e)-[r]-(o:Entity)
    WITH e,
         COUNT(r) as degree,
         COLLECT({
           rel_type: r.type,
           rel_props: properties(r),
           other_node_name: o.name,
           other_node_labels: labels(o)
         })[..10] AS relationships // Limit relationships for context brevity
    RETURN
      e.id AS id,
      e.name AS name,
      labels(e) AS labels,
      properties(e) AS properties,
      degree,
      relationships
    """
    params = {"entity_id": entity_id}
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        return results[0] if results else None
    except Exception as e:
        logger.error(
            f"Error getting context for entity resolution (id: {entity_id}): {e}",
            exc_info=True,
        )
        return None


async def merge_entities(target_id: str, source_id: str) -> bool:
    """
    Merges one entity (source) into another (target) using APOC procedures.
    The source node will be deleted after its relationships are moved.
    """
    query = """
    MATCH (target:Entity {id: $target_id}), (source:Entity {id: $source_id})
    CALL apoc.refactor.mergeNodes([source], target, {
      properties: 'combine',
      mergeRels: true
    }) YIELD node
    RETURN node
    """
    params = {"target_id": target_id, "source_id": source_id}
    try:
        await neo4j_manager.execute_write_query(query, params)
        logger.info(f"Successfully merged node {source_id} into {target_id}.")
        return True
    except Exception as e:
        if "apoc.refactor.mergeNodes" in str(e):
            logger.error(
                "APOC Library not found or configured in Neo4j. "
                "Cannot merge entities. Please install the APOC plugin."
            )
        else:
            logger.error(
                f"Error merging entities ({source_id} -> {target_id}): {e}",
                exc_info=True,
            )
        return False
