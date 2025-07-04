# reset_neo4j.py
import argparse
import asyncio
import time
from typing import Any

import structlog
from config import settings
from core.db_manager import Neo4jManagerSingleton

# Configure a basic logger for this script
# Standard structlog configuration will be handled by config.py if this script is run as part of the app.
# For standalone, basic logging is fine or could be enhanced.
logger = structlog.get_logger(__name__)


# Create an instance of the manager to use its methods
neo4j_manager_instance = Neo4jManagerSingleton()


async def _connect_with_retries(
    uri: str, max_retries: int = 5, retry_delay: int = 5
) -> None:
    """Connect to Neo4j with retry logic."""
    for attempt in range(max_retries):
        try:
            logger.info(
                "Connecting to Neo4j database at %s (Attempt %s/%s)...",
                uri,
                attempt + 1,
                max_retries,
            )
            await neo4j_manager_instance.connect()
            logger.info("Successfully connected to Neo4j.")
            return
        except Exception as exc:
            if attempt < max_retries - 1:
                logger.warning(
                    "Connection failed: %s. Retrying in %s seconds...", exc, retry_delay
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error("Could not connect to Neo4j after multiple retries.")
                raise


async def _delete_all_nodes() -> int:
    """Delete all nodes and relationships from the database."""
    nodes_deleted_total = 0
    while True:
        async with neo4j_manager_instance.driver.session(
            database=settings.NEO4J_DATABASE
        ) as session:  # type: ignore
            tx = await session.begin_transaction()
            result = await tx.run(
                "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) as deleted_nodes_batch"
            )
            single_result_batch = await result.single()
            await tx.commit()
            deleted_in_batch = (
                single_result_batch["deleted_nodes_batch"] if single_result_batch else 0
            )
            nodes_deleted_total += deleted_in_batch
            if deleted_in_batch == 0:
                break
            logger.info("   Deleted %s nodes in this batch...", deleted_in_batch)
    return nodes_deleted_total


async def _drop_all_constraints() -> None:
    """Drop all user-defined constraints."""
    async with neo4j_manager_instance.driver.session(
        database=settings.NEO4J_DATABASE
    ) as session:  # type: ignore
        constraints_result = await session.run("SHOW CONSTRAINTS YIELD name")
        constraints_to_drop = [
            record["name"]
            for record in await constraints_result.data()
            if record["name"]
        ]
        if not constraints_to_drop:
            logger.info("   No user-defined constraints found to drop.")
            return
        for constraint_name in constraints_to_drop:
            try:
                logger.info("   Attempting to drop constraint: %s", constraint_name)
                tx = await session.begin_transaction()
                await tx.run(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                await tx.commit()
                logger.info(
                    "      Dropped constraint '%s' (or it didn't exist).",
                    constraint_name,
                )
            except Exception as exc:
                if tx and not tx.closed():  # type: ignore
                    await tx.rollback()
                logger.warning(
                    "   Note: Could not drop constraint '%s': %s", constraint_name, exc
                )


async def _drop_all_indexes() -> None:
    """Drop all user-defined indexes."""
    async with neo4j_manager_instance.driver.session(
        database=settings.NEO4J_DATABASE
    ) as session:  # type: ignore
        indexes_result = await session.run("SHOW INDEXES YIELD name, type")
        indexes_to_drop_info: list[dict[str, Any]] = await indexes_result.data()
        if not indexes_to_drop_info:
            logger.info("   No user-defined indexes found to drop.")
            return
        for index_info in indexes_to_drop_info:
            index_name = index_info.get("name")
            index_type = index_info.get("type", "").upper()
            if (
                index_name
                and "tokenLookup" not in index_name.lower()
                and "system" not in index_type.lower()
            ):
                try:
                    logger.info(
                        "   Attempting to drop index: %s (type: %s)",
                        index_name,
                        index_type,
                    )
                    tx = await session.begin_transaction()
                    await tx.run(f"DROP INDEX {index_name} IF EXISTS")
                    await tx.commit()
                    logger.info(
                        "      Dropped index '%s' (or it didn't exist).", index_name
                    )
                except Exception as exc:
                    if tx and not tx.closed():  # type: ignore
                        await tx.rollback()
                    logger.warning(
                        "   Note: Could not drop index '%s': %s", index_name, exc
                    )
            elif index_name:
                logger.info(
                    "   Skipping potential system/lookup index: %s (type: %s)",
                    index_name,
                    index_type,
                )


async def reset_neo4j_database_async(uri, user, password, confirm=False):
    """
    Asynchronously resets a Neo4j database by:
    1. Removing all nodes and relationships.
    2. Dropping ALL user-defined constraints.
    3. Dropping ALL user-defined indexes.
    """
    if not confirm:
        response = input(
            "⚠️ WARNING: This will delete ALL data, ALL user-defined constraints, and ALL user-defined indexes "
            "in the Neo4j database. This is a destructive operation. Continue? (y/N): "
        )
        if response.lower() not in ["y", "yes"]:
            logger.info("Operation cancelled.")
            return False

    effective_uri = uri or settings.NEO4J_URI
    effective_user = user or settings.NEO4J_USER
    effective_password = password or settings.NEO4J_PASSWORD

    # Store original settings from the global settings object
    original_uri, original_user, original_pass = (
        settings.NEO4J_URI,
        settings.NEO4J_USER,
        settings.NEO4J_PASSWORD,
    )
    # Temporarily override settings in the global settings object for the duration of this script execution
    settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD = (
        effective_uri,
        effective_user,
        effective_password,
    )

    try:
        await _connect_with_retries(effective_uri)

        async with neo4j_manager_instance.driver.session(
            database=settings.NEO4J_DATABASE,
        ) as session:  # type: ignore
            result = await session.run("MATCH (n) RETURN count(n) as count")
            single_result = await result.single()
            node_count = single_result["count"] if single_result else 0
            logger.info(f"Current database has {node_count} nodes.")

        logger.info("Resetting database data (nodes and relationships)...")
        start_time = time.time()
        nodes_deleted_total = await _delete_all_nodes()
        logger.info("   Total %s nodes deleted.", nodes_deleted_total)

        logger.info("Attempting to drop ALL user-defined constraints...")
        await _drop_all_constraints()

        logger.info(
            "Attempting to drop ALL user-defined indexes (excluding system indexes if identifiable)..."
        )
        await _drop_all_indexes()

        elapsed_time = time.time() - start_time
        logger.info(
            f"✅ Database data, all user-defined constraints, and relevant user-defined indexes reset/dropped in {elapsed_time:.2f} seconds."
        )
        logger.info(
            "   The SAGA system will attempt to recreate its necessary schema on the next run."
        )

        return True

    except Exception as e:
        logger.error(f"❌ Error resetting database: {e}", exc_info=True)
        return False

    finally:
        if neo4j_manager_instance.driver:
            await neo4j_manager_instance.close()
            logger.info("Connection closed.")
        # Restore original settings to the global settings object
        settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD = (
            original_uri,
            original_user,
            original_pass,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reset a Neo4j database by removing all data, user constraints, and user indexes."
    )
    parser.add_argument(
        "--uri",
        default=None,
        help=f"Neo4j connection URI (default: {settings.NEO4J_URI})",
    )
    parser.add_argument(
        "--user",
        default=None,
        help=f"Neo4j username (default: {settings.NEO4J_USER})",
    )
    parser.add_argument(
        "--password",
        default=None,
        help=f"Neo4j password (default: {settings.NEO4J_PASSWORD})",
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    asyncio.run(
        reset_neo4j_database_async(
            uri=args.uri, user=args.user, password=args.password, confirm=args.force
        )
    )
