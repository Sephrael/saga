# data_access/plot_queries.py
import logging
from typing import Dict, Any, List, Tuple
import config
from core_db.base_db_manager import neo4j_manager

logger = logging.getLogger(__name__)

async def save_plot_outline_to_db(plot_data: Dict[str, Any]) -> bool:
    logger.info("Saving decomposed plot outline to Neo4j using MERGE...")
    if not plot_data:
        logger.warning("save_plot_outline_to_db: plot_data is empty. Nothing to save.")
        return False

    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    statements: List[Tuple[str, Dict[str, Any]]] = []
    
    # Clear existing plot structure associated with this novel_id
    statements.append((
        """
        MATCH (ni:NovelInfo {id: $novel_id_param})
        OPTIONAL MATCH (ni)-[r_has_pp:HAS_PLOT_POINT]->(pp:PlotPoint)
        DETACH DELETE pp, r_has_pp
        """,
        {"novel_id_param": novel_id}
    ))
    # Delete the NovelInfo node itself to ensure a clean slate for its properties
    statements.append((
        "MATCH (ni:NovelInfo {id: $novel_id_param}) DETACH DELETE ni",
        {"novel_id_param": novel_id}
    ))

    # Create/update NovelInfo node
    novel_props_for_set = {k: v for k, v in plot_data.items() if not isinstance(v, (list, dict)) and v is not None}
    novel_props_for_set['id'] = novel_id
    statements.append((
        "MERGE (ni:NovelInfo {id: $id_val}) SET ni = $props",
        {"id_val": novel_id, "props": novel_props_for_set}
    ))

    # Create/update PlotPoint nodes and relationships
    plot_points_list_data = plot_data.get('plot_points', [])
    if isinstance(plot_points_list_data, list):
        for i, point_desc_str in enumerate(plot_points_list_data):
            if isinstance(point_desc_str, str):
                pp_id = f"{novel_id}_pp_{i+1}"
                pp_props_for_set = {
                    "id": pp_id,
                    "sequence": i + 1,
                    "description": point_desc_str
                }
                statements.append((
                    "MERGE (pp:PlotPoint {id: $id_val}) SET pp = $props",
                    {"id_val": pp_id, "props": pp_props_for_set}
                ))
                statements.append((
                    """
                    MATCH (ni:NovelInfo {id: $novel_id_param})
                    MATCH (pp:PlotPoint {id: $pp_id_val})
                    MERGE (ni)-[:HAS_PLOT_POINT]->(pp)
                    """,
                    {"novel_id_param": novel_id, "pp_id_val": pp_id}
                ))
                if i > 0: # Link to previous plot point
                    prev_pp_id = f"{novel_id}_pp_{i}"
                    statements.append((
                        """
                        MATCH (prev_pp:PlotPoint {id: $prev_pp_id_val})
                        MATCH (curr_pp:PlotPoint {id: $pp_id_val})
                        MERGE (prev_pp)-[:NEXT_PLOT_POINT]->(curr_pp)
                        """,
                        {"prev_pp_id_val": prev_pp_id, "pp_id_val": pp_id}
                    ))
    try:
        await neo4j_manager.execute_cypher_batch(statements)
        logger.info("Successfully saved decomposed plot outline to Neo4j using MERGE.")
        return True
    except Exception as e:
        logger.error(f"Error saving decomposed plot outline with MERGE: {e}", exc_info=True)
        return False

async def get_plot_outline_from_db() -> Dict[str, Any]:
    logger.info("Loading decomposed plot outline from Neo4j...")
    novel_id = config.MAIN_NOVEL_INFO_NODE_ID
    plot_data: Dict[str, Any] = {}

    novel_info_query = "MATCH (ni:NovelInfo {id: $novel_id_param}) RETURN ni"
    result = await neo4j_manager.execute_read_query(novel_info_query, {"novel_id_param": novel_id})

    if not result or not result[0] or not result[0].get('ni'):
        logger.warning(f"No NovelInfo node found with id '{novel_id}'. Returning empty plot outline.")
        return {}

    plot_data.update(result[0]['ni'])
    plot_data.pop('id', None) # Remove internal ID from the returned dict

    plot_points_query = """
    MATCH (ni:NovelInfo {id: $novel_id_param})-[:HAS_PLOT_POINT]->(pp:PlotPoint)
    RETURN pp.description AS description
    ORDER BY pp.sequence ASC
    """
    pp_results = await neo4j_manager.execute_read_query(plot_points_query, {"novel_id_param": novel_id})
    plot_data['plot_points'] = [record['description'] for record in pp_results] if pp_results else []

    logger.info("Successfully loaded and recomposed plot outline from Neo4j.")
    return plot_data