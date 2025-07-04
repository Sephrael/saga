# data_access/plot_queries.py
from collections.abc import Iterable
from typing import Any

import structlog
from config import settings
from core.db_manager import neo4j_manager

logger = structlog.get_logger(__name__)


async def ensure_novel_info() -> None:
    """Create the NovelInfo node if it does not exist."""
    novel_id = settings.MAIN_NOVEL_INFO_NODE_ID
    query = "MATCH (n:NovelInfo:Entity {id: $id}) RETURN n"
    result = await neo4j_manager.execute_read_query(query, {"id": novel_id})
    if not result or not result[0] or not result[0].get("n"):
        await neo4j_manager.execute_write_query(
            """
            MERGE (n:NovelInfo:Entity {id: $id})
                ON CREATE SET n.title = $title, n.created_ts = timestamp()
            """,
            {"id": novel_id, "title": settings.DEFAULT_PLOT_OUTLINE_TITLE},
        )
        logger.info("Created NovelInfo node with id '%s'", novel_id)


def _build_novel_info_statement(
    novel_id: str, plot_data: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Return Cypher statement to merge NovelInfo properties."""
    props = {
        k: v
        for k, v in plot_data.items()
        if not isinstance(v, list | dict) and v is not None and k != "id"
    }
    props["id"] = novel_id
    return (
        """
    MERGE (ni:Entity {id: $id_val})
    ON CREATE SET ni:NovelInfo, ni = $props, ni.created_ts = timestamp()
    ON MATCH SET  ni:NovelInfo, ni = $props, ni.updated_ts = timestamp()
    """,
        {"id_val": novel_id, "props": props},
    )


def _generate_plot_point_ids(novel_id: str, points: Iterable[Any]) -> set[str]:
    """Return set of PlotPoint IDs expected from ``points``."""
    ids: set[str] = set()
    for i, _ in enumerate(points):
        ids.add(f"pp_{novel_id}_{i + 1}")
    return ids


async def _get_existing_plot_point_ids(novel_id: str) -> set[str]:
    """Retrieve existing PlotPoint IDs from the database."""
    query = (
        "MATCH (:NovelInfo:Entity {id: $novel_id_param})-[:HAS_PLOT_POINT]->(pp:PlotPoint:Entity) "
        "RETURN pp.id AS id"
    )
    result = await neo4j_manager.execute_read_query(query, {"novel_id_param": novel_id})
    return {record["id"] for record in result if record and record.get("id")}


def _build_delete_statements(
    pp_to_delete: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    """Return statements to delete orphaned plot points."""
    if not pp_to_delete:
        return []
    return [
        (
            """
        MATCH (pp:PlotPoint:Entity)
        WHERE pp.id IN $pp_ids_to_delete
        DETACH DELETE pp
        """,
            {"pp_ids_to_delete": list(pp_to_delete)},
        )
    ]


def _build_plot_point_statements(
    novel_id: str, points: Iterable[Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Return statements to merge PlotPoint nodes and relationships."""
    stmts: list[tuple[str, dict[str, Any]]] = []
    for i, point in enumerate(points):
        pp_id = f"pp_{novel_id}_{i + 1}"
        pp_props: dict[str, Any] = {"id": pp_id, "sequence": i + 1, "status": "pending"}
        if isinstance(point, str):
            pp_props["description"] = point
        elif isinstance(point, dict):
            pp_props["description"] = str(point.get("description", ""))
            pp_props["status"] = str(point.get("status", "pending"))
            for k_pp, v_pp in point.items():
                if isinstance(v_pp, str | int | float | bool) and k_pp not in pp_props:
                    pp_props[k_pp] = v_pp
        else:
            logger.warning(f"Skipping invalid plot point item at index {i}: {point}")
            continue
        stmts.append(
            (
                """
            MERGE (pp:Entity {id: $id_val})
            ON CREATE SET pp:PlotPoint, pp = $props, pp.created_ts = timestamp()
            ON MATCH SET  pp:PlotPoint, pp = $props, pp.updated_ts = timestamp()
            """,
                {"id_val": pp_id, "props": pp_props},
            )
        )
        stmts.append(
            (
                """
            MATCH (ni:NovelInfo:Entity {id: $novel_id_param})
            MATCH (pp:PlotPoint:Entity {id: $pp_id_val})
            MERGE (ni)-[:HAS_PLOT_POINT]->(pp)
            """,
                {"novel_id_param": novel_id, "pp_id_val": pp_id},
            )
        )
        if i > 0:
            prev_pp_id = f"pp_{novel_id}_{i}"
            stmts.append(
                (
                    """
                MATCH (prev_pp:PlotPoint:Entity {id: $prev_pp_id_val})
                MATCH (curr_pp:PlotPoint:Entity {id: $curr_pp_id_val})
                OPTIONAL MATCH (prev_pp)-[old_next_rel:NEXT_PLOT_POINT]->(:PlotPoint)
                WHERE old_next_rel IS NOT NULL
                DELETE old_next_rel
                MERGE (prev_pp)-[:NEXT_PLOT_POINT]->(curr_pp)
                """,
                    {"prev_pp_id_val": prev_pp_id, "curr_pp_id_val": pp_id},
                )
            )
    return stmts


async def save_plot_outline_to_db(plot_data: dict[str, Any]) -> bool:
    """Persist the plot outline structure to Neo4j."""

    logger.info("Synchronizing plot outline to Neo4j (non-destructive)...")
    if not plot_data:
        logger.warning(
            "save_plot_outline_to_db: plot_data is empty. No changes will be made."
        )
        return True  # Or False if an empty plot_data implies deletion of existing plot

    novel_id = settings.MAIN_NOVEL_INFO_NODE_ID
    statements: list[tuple[str, dict[str, Any]]] = []

    statements.append(_build_novel_info_statement(novel_id, plot_data))
    plot_points = plot_data.get("plot_points", []) or []
    all_input_pp_ids = _generate_plot_point_ids(novel_id, plot_points)

    try:
        existing_db_pp_ids = await _get_existing_plot_point_ids(novel_id)
    except Exception as e:
        logger.error(
            f"Failed to retrieve existing PlotPoint IDs for novel {novel_id}: {e}",
            exc_info=True,
        )
        return False

    statements.extend(_build_delete_statements(existing_db_pp_ids - all_input_pp_ids))
    statements.extend(_build_plot_point_statements(novel_id, plot_points))

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info(
            f"Successfully synchronized plot outline for novel '{novel_id}' to Neo4j."
        )
        return True
    except Exception as e:
        logger.error(
            f"Error synchronizing plot outline for novel '{novel_id}': {e}",
            exc_info=True,
        )
        return False


async def get_plot_outline_from_db() -> dict[str, Any]:
    """Retrieve the plot outline and associated plot points from Neo4j."""

    logger.info("Loading decomposed plot outline from Neo4j...")
    novel_id = settings.MAIN_NOVEL_INFO_NODE_ID
    plot_data: dict[str, Any] = {}

    # Fetch NovelInfo node properties
    novel_info_query = "MATCH (ni:NovelInfo:Entity {id: $novel_id_param}) RETURN ni"
    result_list = await neo4j_manager.execute_read_query(
        novel_info_query, {"novel_id_param": novel_id}
    )

    if not result_list or not result_list[0] or not result_list[0].get("ni"):
        logger.warning(
            f"No NovelInfo node found with id '{novel_id}'. Returning empty plot outline."
        )
        return {}

    novel_node = result_list[0]["ni"]
    plot_data.update(dict(novel_node))
    plot_data.pop("id", None)  # Remove internal DB ID from returned dict
    plot_data.pop("created_ts", None)
    plot_data.pop("updated_ts", None)

    # Fetch PlotPoints linked to this NovelInfo, ordered by sequence
    plot_points_query = """
    MATCH (ni:NovelInfo:Entity {id: $novel_id_param})-[:HAS_PLOT_POINT]->(pp:PlotPoint:Entity)
    RETURN pp
    ORDER BY pp.sequence ASC
    """
    pp_results = await neo4j_manager.execute_read_query(
        plot_points_query, {"novel_id_param": novel_id}
    )

    # Store plot points. Current orchestrator expects a list of strings (descriptions).
    # If more structured plot point data is needed later, this can be changed to list of dicts.
    fetched_plot_points = []
    if pp_results:
        for record in pp_results:
            pp_node = record.get("pp")
            if pp_node:
                # For compatibility with Orchestrator expecting list of strings:
                fetched_plot_points.append(
                    pp_node.get(
                        "description",
                        f"Plot Point {pp_node.get('sequence')}: Desc missing",
                    )
                )
                # If full plot point data is needed:
                # pp_data = dict(pp_node)
                # pp_data.pop('id', None); pp_data.pop('created_ts', None); pp_data.pop('updated_ts', None)
                # fetched_plot_points.append(pp_data)
    plot_data["plot_points"] = fetched_plot_points

    logger.info(
        f"Successfully loaded plot outline for novel '{novel_id}'. Plot points: {len(plot_data.get('plot_points', []))}"
    )
    return plot_data


async def append_plot_point(description: str, prev_plot_point_id: str) -> str:
    """Append a new PlotPoint node linked to NovelInfo and previous PlotPoint."""
    novel_id = settings.MAIN_NOVEL_INFO_NODE_ID
    # Determine next sequence number
    query = (
        "MATCH (:NovelInfo:Entity {id: $novel_id})-[:HAS_PLOT_POINT]->(pp:PlotPoint:Entity) "
        "RETURN coalesce(max(pp.sequence), 0) AS max_seq"
    )
    result = await neo4j_manager.execute_read_query(query, {"novel_id": novel_id})
    max_seq = result[0].get("max_seq") if result else 0
    next_seq = (max_seq or 0) + 1
    pp_id = f"pp_{novel_id}_{next_seq}"

    statements: list[tuple[str, dict[str, Any]]] = [
        (
            """
        MERGE (pp:Entity {id: $pp_id})
        ON CREATE SET pp:PlotPoint, pp.description = $desc, pp.sequence = $seq, pp.status = 'pending', pp.created_ts = timestamp()
        ON MATCH SET  pp:PlotPoint, pp.description = $desc, pp.sequence = $seq, pp.updated_ts = timestamp()
        """,
            {"pp_id": pp_id, "desc": description, "seq": next_seq},
        ),
        (
            """
        MATCH (ni:NovelInfo:Entity {id: $novel_id})
        MATCH (pp:PlotPoint:Entity {id: $pp_id})
        MERGE (ni)-[:HAS_PLOT_POINT]->(pp)
        """,
            {"novel_id": novel_id, "pp_id": pp_id},
        ),
    ]

    if prev_plot_point_id:
        statements.append(
            (
                """
            MATCH (prev:PlotPoint:Entity {id: $prev_id})
            MATCH (curr:PlotPoint:Entity {id: $pp_id})
            OPTIONAL MATCH (prev)-[r:NEXT_PLOT_POINT]->(:PlotPoint)
            DELETE r
            MERGE (prev)-[:NEXT_PLOT_POINT]->(curr)
            """,
                {"prev_id": prev_plot_point_id, "pp_id": pp_id},
            )
        )

    await neo4j_manager.execute_cypher_batch(statements)
    return pp_id


async def plot_point_exists(description: str) -> bool:
    """Check if a plot point with the given description exists."""
    query = """
    MATCH (pp:PlotPoint:Entity)
    WHERE toLower(pp.description) = toLower($desc)
    RETURN count(pp) AS cnt
    """
    result = await neo4j_manager.execute_read_query(query, {"desc": description})
    return bool(result and result[0] and result[0].get("cnt", 0) > 0)


async def get_last_plot_point_id() -> str | None:
    """Return the ID of the most recent PlotPoint."""
    query = """
    MATCH (pp:PlotPoint:Entity)
    RETURN pp.id AS id
    ORDER BY pp.sequence DESC
    LIMIT 1
    """
    result = await neo4j_manager.execute_read_query(query)
    return result[0].get("id") if result and result[0] else None


async def mark_plot_point_completed(plot_point_index: int) -> None:
    """Set the ``status`` property of a plot point to ``"completed"``."""

    novel_id = settings.MAIN_NOVEL_INFO_NODE_ID
    pp_id = f"pp_{novel_id}_{plot_point_index + 1}"
    query = """
    MATCH (pp:PlotPoint:Entity {id: $pp_id})
    SET pp.status = 'completed', pp.completed_ts = timestamp()
    """
    await neo4j_manager.execute_write_query(query, {"pp_id": pp_id})


async def get_completed_plot_points() -> list[str]:
    """Return descriptions of plot points marked as completed."""

    novel_id = settings.MAIN_NOVEL_INFO_NODE_ID
    query = """
    MATCH (:NovelInfo:Entity {id: $novel_id})-[:HAS_PLOT_POINT]->(pp:PlotPoint:Entity)
    WHERE pp.status = 'completed'
    RETURN pp.description AS desc
    ORDER BY pp.sequence ASC
    """
    result = await neo4j_manager.execute_read_query(query, {"novel_id": novel_id})
    return [r.get("desc") for r in result if r and r.get("desc")]
