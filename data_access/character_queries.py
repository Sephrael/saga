# data_access/character_queries.py
import logging
from typing import Dict, Any, List, Tuple, Optional
import config
from core_db.base_db_manager import neo4j_manager

logger = logging.getLogger(__name__)

async def save_character_profiles_to_db(profiles_data: Dict[str, Any]) -> bool:
    logger.info("Saving decomposed character profiles to Neo4j using MERGE...")
    if not profiles_data:
        logger.warning("save_character_profiles_to_db: profiles_data is empty. Nothing to save.")
        return False

    statements: List[Tuple[str, Dict[str, Any]]] = []

    # Comprehensive clearing of Character related data before rebuilding
    # This approach is simpler than trying to update in place for complex nested structures
    # Remove specific relationships first
    statements.append(("MATCH (c:Character)-[r:HAS_TRAIT]->() DELETE r", {}))
    statements.append(("MATCH (c:Character)-[r:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent) DELETE r, dev", {}))
    statements.append(("MATCH (c1:Character)-[r:DYNAMIC_REL]-(c2:Character) DELETE r", {}))
    # Detach and delete Character nodes (this also removes the :Entity label if it's the only one left after removing :Character)
    statements.append(("MATCH (c:Character) DETACH DELETE c", {})) # Should also delete :Entity if it only has :Character
    # Delete Trait nodes
    statements.append(("MATCH (t:Trait) DETACH DELETE t", {}))


    for char_name, profile in profiles_data.items():
        if not isinstance(profile, dict):
            logger.warning(f"Skipping invalid profile for '{char_name}' (not a dict).")
            continue

        char_props_for_set = {k: v for k, v in profile.items() if isinstance(v, (str, int, float, bool)) and v is not None}
        
        # Merge character node, ensuring :Entity and :Character labels
        character_node_query = """
        MERGE (c:Entity {name: $char_name_val}) 
        SET c:Character                         // Ensure :Character label is present
        SET c += $props                         // Set/update properties
        """
        statements.append((
            character_node_query, 
            {"char_name_val": char_name, "props": char_props_for_set}
        ))

        # Traits
        if isinstance(profile.get("traits"), list):
            for trait_str in profile["traits"]:
                if isinstance(trait_str, str):
                    statements.append((
                        """
                        MATCH (c:Character:Entity {name: $char_name_val}) 
                        MERGE (t:Trait {name: $trait_name_val})
                        MERGE (c)-[:HAS_TRAIT]->(t)
                        """,
                        {"char_name_val": char_name, "trait_name_val": trait_str}
                    ))
        
        # Relationships (DYNAMIC_REL)
        if isinstance(profile.get("relationships"), dict):
            for target_char_name, rel_detail in profile["relationships"].items():
                rel_type_str = "RELATED_TO" # Default relationship type
                rel_props_for_set = {"description": str(rel_detail)} # Default props

                if isinstance(rel_detail, dict): # If relationship detail is a dict itself
                    rel_type_str = str(rel_detail.get("type", rel_type_str)).upper().replace(" ", "_")
                    rel_props_for_set = {k:v for k,v in rel_detail.items() if isinstance(v, (str, int, float, bool))}
                    rel_props_for_set.setdefault("description", f"{rel_type_str} {target_char_name}") # Ensure description
                elif isinstance(rel_detail, str): # If relationship detail is just a string (treat as type)
                    rel_type_str = rel_detail.upper().replace(" ", "_")
                    rel_props_for_set = {"description": rel_detail}
                
                # Add chapter_added and is_provisional from profile metadata
                # Assuming KG_PREPOPULATION_CHAPTER_NUM is 0 for initial setup
                source_chapter_key = f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}"
                rel_props_for_set.setdefault("chapter_added", profile.get(source_chapter_key, config.KG_PREPOPULATION_CHAPTER_NUM))
                rel_props_for_set.setdefault("is_provisional", profile.get(source_chapter_key) == "provisional_from_unrevised_draft")

                statements.append((
                    """
                    MATCH (c1:Character:Entity {name: $char_name1_val})
                    MERGE (c2:Entity {name: $char_name2_val}) // Target can be any entity, ensure :Character if relationship implies
                        ON CREATE SET c2:Character, c2.description = 'Placeholder desc - created via rel from ' + $char_name1_val // Auto-create target as Character if new
                        ON MATCH SET c2:Character // Ensure existing target also gets :Character label
                    MERGE (c1)-[r:DYNAMIC_REL {type: $rel_type_val}]->(c2)
                    SET r += $rel_props_val
                    """,
                    {
                        "char_name1_val": char_name,
                        "char_name2_val": target_char_name,
                        "rel_type_val": rel_type_str,
                        "rel_props_val": rel_props_for_set
                    }
                ))

        # Development Events
        for key, value_str in profile.items():
            if key.startswith("development_in_chapter_") and isinstance(value_str, str):
                try:
                    chap_num_int = int(key.split("_")[-1])
                    dev_event_props = {
                        "summary": value_str,
                        "chapter_updated": chap_num_int
                        # Add provisional status from profile metadata if available
                    }
                    provisional_dev_key = f"source_quality_chapter_{chap_num_int}"
                    if profile.get(provisional_dev_key) == "provisional_from_unrevised_draft":
                        dev_event_props["is_provisional"] = True
                    
                    statements.append((
                        """
                        MATCH (c:Character:Entity {name: $char_name_val}) 
                        CREATE (dev:DevelopmentEvent)
                        SET dev = $props
                        CREATE (c)-[:DEVELOPED_IN_CHAPTER]->(dev)
                        """,
                        {"char_name_val": char_name, "props": dev_event_props}
                    ))
                except ValueError:
                    logger.warning(f"Could not parse chapter number from development key: {key} for char {char_name}")
    try:
        await neo4j_manager.execute_cypher_batch(statements)
        logger.info("Successfully saved decomposed character profiles to Neo4j using MERGE.")
        return True
    except Exception as e:
        logger.error(f"Error saving decomposed character profiles with MERGE: {e}", exc_info=True)
        return False

async def get_character_profiles_from_db() -> Dict[str, Any]:
    logger.info("Loading decomposed character profiles from Neo4j...")
    profiles_data: Dict[str, Any] = {}

    char_query = "MATCH (c:Character:Entity) RETURN c" # Ensure both labels are present
    char_results = await neo4j_manager.execute_read_query(char_query)

    if not char_results:
        return {}

    for record in char_results:
        char_node = record['c']
        char_name = char_node.get('name')
        if not char_name:
            continue # Skip nodes without a name

        profile = dict(char_node)
        profile.pop('name', None) # Name is key, not property in dict

        # Fetch traits
        traits_query = "MATCH (:Character:Entity {name: $char_name})-[:HAS_TRAIT]->(t:Trait) RETURN t.name AS trait_name"
        trait_results = await neo4j_manager.execute_read_query(traits_query, {"char_name": char_name})
        profile["traits"] = [tr['trait_name'] for tr in trait_results] if trait_results else []

        # Fetch relationships
        rels_query = """
        MATCH (:Character:Entity {name: $char_name})-[r:DYNAMIC_REL]->(target:Entity) // Target can be any Entity
        RETURN target.name AS target_name, r.type AS relationship_type, properties(r) AS rel_props
        """
        rel_results = await neo4j_manager.execute_read_query(rels_query, {"char_name": char_name})
        relationships = {}
        if rel_results:
            for rel_rec in rel_results:
                target_name = rel_rec['target_name']
                rel_props = rel_rec.get('rel_props', {}) # Ensure rel_props is a dict
                # rel_type is already a property in rel_props due to {type: $rel_type_val} in MERGE
                # but if not, we can get it from rel_rec['relationship_type']
                if 'type' not in rel_props and 'relationship_type' in rel_rec:
                    rel_props['type'] = rel_rec['relationship_type']
                relationships[target_name] = rel_props

        profile["relationships"] = relationships

        # Fetch development events
        dev_query = """
        MATCH (:Character:Entity {name: $char_name})-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)
        RETURN dev.summary AS summary, dev.chapter_updated AS chapter, dev.is_provisional AS is_provisional
        """
        dev_results = await neo4j_manager.execute_read_query(dev_query, {"char_name": char_name})
        if dev_results:
            for dev_rec in dev_results:
                dev_key = f"development_in_chapter_{dev_rec['chapter']}"
                profile[dev_key] = dev_rec['summary']
                if dev_rec.get('is_provisional'): # Store provisional status in profile if present
                    profile[f"source_quality_chapter_{dev_rec['chapter']}"] = "provisional_from_unrevised_draft"
        
        profiles_data[char_name] = profile

    logger.info(f"Successfully loaded and recomposed {len(profiles_data)} character profiles from Neo4j.")
    return profiles_data

async def get_character_info_for_snippet_from_db(char_name: str, chapter_limit: int) -> Optional[Dict[str, Any]]:
    # This query fetches the character's description and status directly.
    # For the most recent development note, it finds the DevelopmentEvent with the highest chapter_updated <= chapter_limit.
    # It also checks for any provisional data associated with the character up to chapter_limit.
    query = """
    MATCH (c:Character:Entity {name: $char_name_param})
    
    // Get latest non-provisional development note if available
    OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev_np:DevelopmentEvent)
    WHERE dev_np.chapter_updated <= $chapter_limit_param AND (dev_np.is_provisional IS NULL OR dev_np.is_provisional = FALSE)
    WITH c, dev_np ORDER BY dev_np.chapter_updated DESC
    WITH c, HEAD(COLLECT(dev_np)) AS latest_non_provisional_dev_event

    // Get latest provisional development note if available (and more recent than non-provisional)
    OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev_p:DevelopmentEvent)
    WHERE dev_p.chapter_updated <= $chapter_limit_param AND dev_p.is_provisional = TRUE
    WITH c, latest_non_provisional_dev_event, dev_p ORDER BY dev_p.chapter_updated DESC
    WITH c, latest_non_provisional_dev_event, HEAD(COLLECT(dev_p)) AS latest_provisional_dev_event

    // Determine which development event to use
    WITH c,
         CASE
           WHEN latest_provisional_dev_event IS NOT NULL AND (latest_non_provisional_dev_event IS NULL OR latest_provisional_dev_event.chapter_updated >= latest_non_provisional_dev_event.chapter_updated) THEN latest_provisional_dev_event
           ELSE latest_non_provisional_dev_event
         END AS latest_dev_event

    // Check for any provisional data markers on the character node itself or related DYNAMIC_RELs or DevelopmentEvents
    OPTIONAL MATCH (c)-[prov_rel:DYNAMIC_REL]-(:Entity)
    WHERE prov_rel.chapter_added <= $chapter_limit_param AND prov_rel.is_provisional = TRUE
    
    OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(any_prov_dev:DevelopmentEvent)
    WHERE any_prov_dev.chapter_updated <= $chapter_limit_param AND any_prov_dev.is_provisional = TRUE
    
    RETURN c.description AS description,
           c.status AS current_status,
           latest_dev_event.summary AS most_recent_development_note,
           (c.is_provisional = TRUE OR prov_rel IS NOT NULL OR any_prov_dev IS NOT NULL) AS is_provisional_overall
    LIMIT 1
    """
    params = {"char_name_param": char_name, "chapter_limit_param": chapter_limit}
    try:
        result = await neo4j_manager.execute_read_query(query, params)
        if result and result[0]:
            record = result[0]
            # Ensure development note is "N/A" if null, rather than None object
            dev_note = record.get("most_recent_development_note")
            if dev_note is None: dev_note = "N/A"

            return {
                "description": record.get("description"),
                "current_status": record.get("current_status"),
                "most_recent_development_note": dev_note,
                "is_provisional_overall": record.get("is_provisional_overall", False)
            }
        logger.debug(f"No detailed snippet info found for character '{char_name}' in Neo4j up to chapter {chapter_limit}.")
    except Exception as e:
        logger.error(f"Error fetching character info for snippet ({char_name}) from Neo4j: {e}", exc_info=True)
    return None