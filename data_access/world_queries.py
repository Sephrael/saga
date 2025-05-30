# data_access/world_queries.py
import logging
from typing import Dict, Any, List, Tuple, Optional
import config
from core_db.base_db_manager import neo4j_manager

logger = logging.getLogger(__name__)

async def save_world_building_to_db(world_data: Dict[str, Any]) -> bool:
    logger.info("Saving decomposed world building data to Neo4j using MERGE...")
    if not world_data:
        logger.warning("save_world_building_to_db: world_data is empty. Nothing to save.")
        return False

    statements: List[Tuple[str, Dict[str, Any]]] = []

    # Clear existing world structure
    # Refined: Changed "MATCH (we:WorldElement) OPTIONAL MATCH (we)-[r]-() DETACH DELETE we, r" to "MATCH (we:WorldElement) DETACH DELETE we"
    # as "DETACH DELETE we, r" is invalid Cypher. DETACH DELETE we handles relationships.
    statements.append(("MATCH (we:WorldElement) DETACH DELETE we", {}))
    statements.append(("MATCH (wev:WorldElaborationEvent) DETACH DELETE wev", {}))
    statements.append(("MATCH (wc:WorldContainer {id: $wc_id_param}) DETACH DELETE wc", {"wc_id_param": config.MAIN_WORLD_CONTAINER_NODE_ID}))
    statements.append(("MATCH (vn:ValueNode) DETACH DELETE vn", {})) # Clear ValueNodes used for list properties


    for category_str, items_dict_value_from_world_data in world_data.items():
        if category_str == "_overview_":
            if isinstance(items_dict_value_from_world_data, dict) and "description" in items_dict_value_from_world_data:
                wc_id = config.MAIN_WORLD_CONTAINER_NODE_ID
                desc_to_set = str(items_dict_value_from_world_data.get("description", ""))
                wc_props = {
                    "id": wc_id,
                    "overview_description": desc_to_set
                }
                # Add provisional status from profile metadata if available
                source_chapter_key = f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}"
                if items_dict_value_from_world_data.get(source_chapter_key) == "provisional_from_unrevised_draft":
                     wc_props["is_provisional"] = True

                statements.append((
                    "MERGE (wc:WorldContainer {id: $id_val}) SET wc = $props",
                    {"id_val": wc_id, "props": wc_props }
                ))
            continue # End _overview_ processing

        # Skip metadata keys or non-dict categories
        if category_str in ["is_default", "source", "user_supplied_data"] or \
           not isinstance(items_dict_value_from_world_data, dict):
            continue
        
        items_category_dict = items_dict_value_from_world_data
        for item_name_str, details_dict in items_category_dict.items():
            if not isinstance(details_dict, dict) or \
               item_name_str.startswith(("_", "source_quality_chapter_", "category_updated_in_chapter_")):
                continue # Skip internal keys or malformed items

            we_id_str = f"{category_str}_{item_name_str}".replace(" ", "_").replace("'", "").lower()
            item_props_for_set = {k: v for k, v in details_dict.items() if isinstance(v, (str, int, float, bool)) and v is not None}
            item_props_for_set['id'] = we_id_str
            item_props_for_set['name'] = item_name_str
            item_props_for_set['category'] = category_str

            # Determine created_chapter and provisional status
            created_chap_num = config.KG_PREPOPULATION_CHAPTER_NUM # Default
            is_item_provisional = False
            added_key = next((k for k in details_dict if k.startswith("added_in_chapter_")), None)
            if added_key:
                try: created_chap_num = int(added_key.split("_")[-1])
                except ValueError: pass # Keep default if parse fails
            
            source_quality_key_for_creation = f"source_quality_chapter_{created_chap_num}"
            if details_dict.get(source_quality_key_for_creation) == "provisional_from_unrevised_draft":
                is_item_provisional = True
            
            item_props_for_set['created_chapter'] = created_chap_num
            if is_item_provisional:
                item_props_for_set['is_provisional'] = True

            statements.append((
                "MERGE (we:WorldElement {id: $id_val}) SET we = $props",
                {"id_val": we_id_str, "props": item_props_for_set}
            ))

            # Handle list properties by creating ValueNode and relationships
            for list_prop_key_str in ["goals", "rules", "key_elements", "traits"]:
                list_value = details_dict.get(list_prop_key_str)
                if isinstance(list_value, list):
                    for val_item_from_list in list_value:
                        if isinstance(val_item_from_list, str):
                            # Determine relationship name based on property key
                            rel_name_internal_str = f"HAS_{list_prop_key_str.upper().rstrip('S')}"
                            if list_prop_key_str == "key_elements": rel_name_internal_str = "HAS_KEY_ELEMENT"
                            elif list_prop_key_str == "traits": rel_name_internal_str = "HAS_TRAIT_ASPECT"
                            
                            statements.append((
                                f"""
                                MATCH (we:WorldElement {{id: $we_id_val}})
                                MERGE (v:ValueNode {{value: $val_item_value, type: $value_node_type}})
                                MERGE (we)-[:{rel_name_internal_str}]->(v)
                                """,
                                {"we_id_val": we_id_str, "val_item_value": val_item_from_list, "value_node_type": list_prop_key_str}
                            ))
            
            # Handle elaboration events
            for key_str, value_val in details_dict.items():
                if key_str.startswith("elaboration_in_chapter_") and isinstance(value_val, str):
                    try:
                        chap_num_val = int(key_str.split("_")[-1])
                        elab_props = {
                            "summary": value_val,
                            "chapter_updated": chap_num_val
                        }
                        provisional_elab_key = f"source_quality_chapter_{chap_num_val}"
                        if details_dict.get(provisional_elab_key) == "provisional_from_unrevised_draft":
                            elab_props["is_provisional"] = True
                        
                        statements.append((
                            f"""
                            MATCH (we:WorldElement {{id: $we_id_val}})
                            CREATE (we_elab:WorldElaborationEvent)
                            SET we_elab = $props
                            CREATE (we)-[:ELABORATED_IN_CHAPTER]->(we_elab)
                            """,
                            {"we_id_val": we_id_str, "props": elab_props}
                        ))
                    except ValueError:
                        logger.warning(f"Could not parse chapter number from world elaboration key: {key_str} for item {item_name_str}")
    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
            logger.info("Successfully saved decomposed world building data to Neo4j using MERGE.")
        else:
            logger.info("No statements generated for saving world building data.")
        return True
    except Exception as e:
        logger.error(f"Error saving decomposed world building data with MERGE: {e}", exc_info=True)
        return False

async def get_world_building_from_db() -> Dict[str, Any]:
    logger.info("Loading decomposed world building data from Neo4j...")
    world_data: Dict[str, Any] = {"_overview_": {}}

    # Load overview
    overview_query = "MATCH (wc:WorldContainer {id: $wc_id_param}) RETURN wc.overview_description AS desc, wc.is_provisional AS is_provisional"
    overview_res = await neo4j_manager.execute_read_query(overview_query, {"wc_id_param": config.MAIN_WORLD_CONTAINER_NODE_ID})
    if overview_res and overview_res[0] and overview_res[0].get('desc') is not None:
        world_data["_overview_"]["description"] = overview_res[0]['desc']
        if overview_res[0].get('is_provisional'): # Load provisional status if present
            world_data["_overview_"][f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}"] = "provisional_from_unrevised_draft"


    # Load WorldElements
    we_query = "MATCH (we:WorldElement) RETURN we"
    we_results = await neo4j_manager.execute_read_query(we_query)
    
    # Initialize standard categories
    standard_categories = ["locations", "society", "systems", "lore", "history", "factions"]
    for cat_key in standard_categories:
        world_data.setdefault(cat_key, {}) # Ensure category key exists

    if not we_results:
        return world_data

    for record in we_results:
        we_node = record['we']
        category = we_node.get('category')
        item_name = we_node.get('name')
        we_id = we_node.get('id') # ID used for fetching related list properties

        if not category or not item_name or not we_id:
            continue

        if category not in world_data: # Should be initialized already, but as a safeguard
            world_data[category] = {}
        
        item_detail = dict(we_node) # Start with all node properties
        item_detail.pop('id', None); item_detail.pop('name', None); item_detail.pop('category', None)

        # Restore added_in_chapter_X key and provisional status if applicable
        created_chapter_num = item_detail.pop('created_chapter', config.KG_PREPOPULATION_CHAPTER_NUM)
        item_detail[f"added_in_chapter_{created_chapter_num}"] = True # Mark as added
        if item_detail.pop('is_provisional', False): # If 'is_provisional' was true on node
            item_detail[f"source_quality_chapter_{created_chapter_num}"] = "provisional_from_unrevised_draft"
        

        # Fetch list properties (goals, rules, key_elements, traits)
        for list_prop_key in ["goals", "rules", "key_elements", "traits"]:
            rel_name_query = f"HAS_{list_prop_key.upper().rstrip('S')}"
            if list_prop_key == "key_elements": rel_name_query = "HAS_KEY_ELEMENT"
            elif list_prop_key == "traits": rel_name_query = "HAS_TRAIT_ASPECT"

            list_values_query = """
            MATCH (:WorldElement {id: $we_id_param})-[:%s]->(v:ValueNode {type: $value_node_type_param})
            RETURN v.value AS item_value
            """ % rel_name_query # Safe to format rel_name_query as it's from a controlled list
            
            list_val_res = await neo4j_manager.execute_read_query(
                list_values_query,
                {"we_id_param": we_id, "value_node_type_param": list_prop_key}
            )
            item_detail[list_prop_key] = [res_item['item_value'] for res_item in list_val_res] if list_val_res else []

        # Fetch elaboration events
        elab_query = """
        MATCH (:WorldElement {id: $we_id_param})-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent)
        RETURN elab.summary AS summary, elab.chapter_updated AS chapter, elab.is_provisional AS is_provisional
        """
        elab_results = await neo4j_manager.execute_read_query(elab_query, {"we_id_param": we_id})
        if elab_results:
            for elab_rec in elab_results:
                elab_key = f"elaboration_in_chapter_{elab_rec['chapter']}"
                item_detail[elab_key] = elab_rec['summary']
                if elab_rec.get('is_provisional'): # Store provisional status from elaboration
                    item_detail[f"source_quality_chapter_{elab_rec['chapter']}"] = "provisional_from_unrevised_draft"
        
        world_data[category][item_name] = item_detail

    logger.info(f"Successfully loaded and recomposed world building data from Neo4j.")
    return world_data

async def get_world_elements_for_snippet_from_db(category: str, chapter_limit: int, item_limit: int) -> List[Dict[str, Any]]:
    # This query fetches WorldElements of a specific category.
    # It checks if the item itself is marked provisional OR if it has any provisional elaborations up to chapter_limit.
    query = """
    MATCH (we:WorldElement {category: $category_param})
    WHERE (we.created_chapter IS NULL OR we.created_chapter <= $chapter_limit_param) // Ensure element itself is relevant up to chapter_limit

    // Check for any provisional elaboration linked to this world element up to the chapter limit
    OPTIONAL MATCH (we)-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent)
    WHERE elab.chapter_updated <= $chapter_limit_param AND elab.is_provisional = TRUE
    
    // Group by world element to determine if it's effectively provisional
    WITH we, COLLECT(DISTINCT elab) AS provisional_elaborations 
    WITH we, ( (we.is_provisional = TRUE AND (we.created_chapter IS NULL OR we.created_chapter <= $chapter_limit_param) ) OR size(provisional_elaborations) > 0) AS is_item_provisional_overall
    
    RETURN we.name AS name,
           we.description AS description, // Fetch full description for snippet creation
           is_item_provisional_overall AS is_provisional
    ORDER BY we.name ASC // Consistent ordering
    LIMIT $item_limit_param
    """
    params = {"category_param": category, "chapter_limit_param": chapter_limit, "item_limit_param": item_limit}
    items = []
    try:
        results = await neo4j_manager.execute_read_query(query, params)
        if results:
            for record in results:
                desc = record.get("description") or ""
                items.append({
                    "name": record.get("name"),
                    # Create snippet from full description
                    "description_snippet": (desc[:50].strip() + "..." if len(desc) > 50 else desc.strip()),
                    "is_provisional": record.get("is_provisional", False)
                })
    except Exception as e:
        logger.error(f"Error fetching world elements for snippet (category {category}) from Neo4j: {e}", exc_info=True)
    return items