# data_access/kg_queries.py
import logging
from typing import Optional, List, Dict, Any, Tuple
import config
from core_db.base_db_manager import neo4j_manager

logger = logging.getLogger(__name__)

async def add_kg_triple_to_db(
    subject: str,
    predicate: str,
    obj_val: str,
    chapter_added: int,
    confidence: float = 1.0,
    is_provisional: bool = False
):
    subj_s, pred_s, obj_s = subject.strip(), predicate.strip(), obj_val.strip()
    if not all([subj_s, pred_s, obj_s]) or chapter_added < config.KG_PREPOPULATION_CHAPTER_NUM:
        logger.warning(f"Neo4j: Invalid KG triple for add: S='{subj_s}', P='{pred_s}', O='{obj_s}', Chap={chapter_added}")
        return

    # Refined: Use direct MERGE with ON CREATE / ON MATCH for relationship upsert
    query = """
    MERGE (s:Entity {name: $subject_param})
    MERGE (o:Entity {name: $object_param})
    MERGE (s)-[r:DYNAMIC_REL {
        type: $predicate_param,
        chapter_added: $chapter_added_param 
    }]->(o)
    ON CREATE SET
        r.is_provisional = $is_provisional_param,
        r.confidence = $confidence_param,
        r.created_at = timestamp(),
        r.last_updated = timestamp()
    ON MATCH SET
        r.is_provisional = $is_provisional_param,
        r.confidence = $confidence_param,
        r.last_updated = timestamp()
    """
    parameters = {
        "subject_param": subj_s,
        "object_param": obj_s,
        "predicate_param": pred_s,
        "chapter_added_param": chapter_added,
        "is_provisional_param": is_provisional,
        "confidence_param": confidence
    }
    try:
        await neo4j_manager.execute_write_query(query, parameters)
        logger.debug(f"Neo4j: Added/Updated KG triple for Ch {chapter_added}: ({subj_s}, {pred_s}, {obj_s}). Prov: {is_provisional}, Conf: {confidence}")
    except Exception as e:
        logger.error(f"Neo4j: Error adding KG triple: ({subj_s}, {pred_s}, {obj_s}). Error: {e}", exc_info=True)

async def add_kg_triples_batch_to_db(
    triples_data: List[Tuple[str, str, str, int, float, bool]] # subj, pred, obj, chapter, confidence, provisional
):
    if not triples_data:
        logger.info("Neo4j: add_kg_triples_batch_to_db: No triples to add.")
        return

    statements_with_params: List[Tuple[str, Dict[str, Any]]] = []
    # This query is identical to the single add, as execute_cypher_batch handles running multiple statements in one transaction.
    base_query = """
    MERGE (s:Entity {name: $subject_param})
    MERGE (o:Entity {name: $object_param})
    MERGE (s)-[r:DYNAMIC_REL {
        type: $predicate_param,
        chapter_added: $chapter_added_param
    }]->(o)
    ON CREATE SET
        r.is_provisional = $is_provisional_param,
        r.confidence = $confidence_param,
        r.created_at = timestamp(),
        r.last_updated = timestamp()
    ON MATCH SET
        r.is_provisional = $is_provisional_param,
        r.confidence = $confidence_param,
        r.last_updated = timestamp()
    """
    for subj, pred, obj_val, chapter_added, confidence, is_provisional in triples_data:
        subj_s, pred_s, obj_s = subj.strip(), pred.strip(), obj_val.strip()
        if not all([subj_s, pred_s, obj_s]) or chapter_added < config.KG_PREPOPULATION_CHAPTER_NUM:
            logger.warning(f"Neo4j (Batch): Invalid KG triple skipped: S='{subj_s}', P='{pred_s}', O='{obj_s}', Chap={chapter_added}")
            continue
        
        parameters = {
            "subject_param": subj_s,
            "object_param": obj_s,
            "predicate_param": pred_s,
            "chapter_added_param": chapter_added,
            "is_provisional_param": is_provisional,
            "confidence_param": confidence
        }
        statements_with_params.append((base_query, parameters))

    if not statements_with_params:
        logger.info("Neo4j: add_kg_triples_batch_to_db: No valid triples to add after filtering.")
        return

    try:
        await neo4j_manager.execute_cypher_batch(statements_with_params)
        logger.info(f"Neo4j: Batch added/updated {len(statements_with_params)} KG triples.")
    except Exception as e:
        # Log first few problematic params for debugging, if any
        first_few_params_str = str([p for _, p in statements_with_params[:3]]) if statements_with_params else "N/A"
        logger.error(f"Neo4j: Error in batch adding KG triples. First few params: {first_few_params_str}. Error: {e}", exc_info=True)
        raise # Re-raise to inform the caller

async def query_kg_from_db(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    obj_val: Optional[str] = None,
    chapter_limit: Optional[int] = None,
    include_provisional: bool = True,
    limit_results: Optional[int] = None
) -> List[Dict[str, Any]]:
    conditions = []
    parameters: Dict[str, Any] = {} # Explicitly type parameters
    match_clause = "MATCH (s:Entity)-[r:DYNAMIC_REL]->(o:Entity)"

    if subject is not None:
        conditions.append("s.name = $subject_param")
        parameters["subject_param"] = subject.strip()
    if predicate is not None:
        conditions.append("r.type = $predicate_param")
        parameters["predicate_param"] = predicate.strip()
    if obj_val is not None:
        conditions.append("o.name = $object_param")
        parameters["object_param"] = obj_val.strip()
    if chapter_limit is not None:
        conditions.append("r.chapter_added <= $chapter_limit_param")
        parameters["chapter_limit_param"] = chapter_limit
    if not include_provisional:
        # Check for explicit FALSE, or if the property doesn't exist (older data might not have it)
        conditions.append("(r.is_provisional = FALSE OR r.is_provisional IS NULL)")


    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    return_clause = """
    RETURN s.name AS subject, r.type AS predicate, o.name AS object,
           r.chapter_added AS chapter_added, r.confidence AS confidence, r.is_provisional AS is_provisional
    """
    order_clause = " ORDER BY r.chapter_added DESC, r.confidence DESC"
    limit_clause_str = "" # Renamed to avoid conflict
    if limit_results is not None and limit_results > 0:
        limit_clause_str = f" LIMIT {int(limit_results)}"
    
    full_query = match_clause + where_clause + return_clause + order_clause + limit_clause_str
    try:
        results = await neo4j_manager.execute_read_query(full_query, parameters)
        triples_list: List[Dict[str, Any]] = [dict(record) for record in results] if results else []
        logger.debug(f"Neo4j: KG query returned {len(triples_list)} results for query: {full_query} with params {parameters}")
        return triples_list
    except Exception as e:
        logger.error(f"Neo4j: Error querying KG. Query: {full_query}, Params: {parameters}, Error: {e}", exc_info=True)
        return []

async def get_most_recent_value_from_db(
    subject: str,
    predicate: str,
    chapter_limit: Optional[int] = None,
    include_provisional: bool = False # Default changed to False to prefer canon facts
) -> Optional[str]:
    if not subject.strip() or not predicate.strip():
        logger.warning(f"Neo4j: get_most_recent_value_from_db: empty subject or predicate. S='{subject}', P='{predicate}'")
        return None

    results = await query_kg_from_db(
        subject=subject,
        predicate=predicate,
        chapter_limit=chapter_limit,
        include_provisional=include_provisional,
        limit_results=1 # We only need the most recent one
    )
    if results and results[0] and 'object' in results[0]:
        value = str(results[0]["object"])
        logger.debug(f"Neo4j: Found most recent value for ('{subject}', '{predicate}'): '{value}' from Ch {results[0].get('chapter_added','N/A')}, Provisional included: {include_provisional}, Actual provisional status: {results[0].get('is_provisional')}")
        return value
    
    # If no result with specified provisionality, try the other way if include_provisional was specific
    if include_provisional is False: # We explicitly asked for non-provisional and found nothing
        logger.debug(f"Neo4j: No non-provisional value for ({subject}, {predicate}) up to Ch {chapter_limit}. Trying with provisional if allowed by broader context.")
        # This function call is specific, the caller might decide to try again with include_provisional=True
    elif include_provisional is True and not results: # We asked for provisional included and found nothing at all
         logger.debug(f"Neo4j: No value (provisional or non-provisional) for ({subject}, {predicate}) up to Ch {chapter_limit}")


    logger.debug(f"Neo4j: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}, with include_provisional={include_provisional}.")
    return None