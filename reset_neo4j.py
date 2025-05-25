# reset_neo4j.py
import argparse
import time
from core_db.base_db_manager import Neo4jManagerSingleton # Use the singleton
import config # To get default URI, user, pass if not provided via args
import asyncio # Required to call async methods

# Create an instance of the manager to use its methods
# This is okay for a script like this, as it will be a single instance for the script's lifetime.
neo4j_manager_instance = Neo4jManagerSingleton()


async def reset_neo4j_database_async(uri, user, password, confirm=False):
    """
    Asynchronously resets a Neo4j database by removing all nodes and relationships.
    """
    if not confirm:
        response = input("⚠️ WARNING: This will delete ALL data in the Neo4j database. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return False
    
    # Override config if args are provided
    effective_uri = uri or config.NEO4J_URI
    effective_user = user or config.NEO4J_USER
    effective_password = password or config.NEO4J_PASSWORD

    # Temporarily set config for the neo4j_manager if args differ, or ensure it uses them
    # A cleaner way would be to pass these to connect() if it supported it,
    # or have a temporary context for the manager. For this script, direct override is simpler.
    original_uri, original_user, original_pass = config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD
    config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD = effective_uri, effective_user, effective_password

    try:
        print(f"Connecting to Neo4j database at {effective_uri}...")
        await neo4j_manager_instance.connect() # Connects using current config values

        async with neo4j_manager_instance.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
            # Test connection and get initial count
            result = await session.run("MATCH (n) RETURN count(n) as count")
            single_result = await result.single()
            node_count = single_result["count"] if single_result else 0
            print(f"Current database has {node_count} nodes.")
        
        print("Resetting database...")
        start_time = time.time()
        
        # Batching delete operations might be more performant on very large databases
        # but for a full reset, separate deletes are clear.
        # Consider `CALL apoc.periodic.iterate` for huge databases if APOC is available.
        
        # Detach delete all nodes (which also deletes their relationships)
        # This is generally more robust than deleting relationships then nodes separately.
        # Using batches to avoid overly large transactions if the DB is huge.
        while True:
            async with neo4j_manager_instance.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
                result = await session.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) as deleted_nodes_batch")
                single_result_batch = await result.single()
                deleted_in_batch = single_result_batch["deleted_nodes_batch"] if single_result_batch else 0
                if deleted_in_batch == 0:
                    break # No more nodes to delete
                print(f"   Deleted {deleted_in_batch} nodes in this batch...")
        
        # Additionally, clear out any remaining schema like indexes (constraints are harder to drop programmatically without knowing names)
        # For a true "reset to factory", one might need to drop and recreate the database itself,
        # but DETACH DELETE n handles all data.
        # Dropping indexes and constraints might be desired for a full reset.
        # Example: Dropping a specific known index (vector index)
        try:
            async with neo4j_manager_instance.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
                await session.run(f"DROP INDEX {config.NEO4J_VECTOR_INDEX_NAME} IF EXISTS")
                print(f"Dropped vector index '{config.NEO4J_VECTOR_INDEX_NAME}' if it existed.")
        except Exception as e_index:
            print(f"Note: Could not drop vector index '{config.NEO4J_VECTOR_INDEX_NAME}' (may not exist or other issue): {e_index}")


        elapsed_time = time.time() - start_time
        print(f"✅ Database data reset complete in {elapsed_time:.2f} seconds.")
        print(f"   Note: Constraints might need manual dropping/recreation if a full schema reset is desired.")
        print(f"   The SAGA system will attempt to recreate necessary constraints and indexes on its next run.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        return False
    
    finally:
        if neo4j_manager_instance.driver:
            await neo4j_manager_instance.close()
            print("Connection closed.")
        # Restore original config values if they were changed
        config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD = original_uri, original_user, original_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset a Neo4j database by removing all data (nodes and relationships).")
    parser.add_argument("--uri", default=None, help=f"Neo4j connection URI (default: {config.NEO4J_URI})")
    parser.add_argument("--user", default=None, help=f"Neo4j username (default: {config.NEO4J_USER})")
    parser.add_argument("--password", default=None, help=f"Neo4j password (default: {config.NEO4J_PASSWORD})")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(reset_neo4j_database_async(
        uri=args.uri,
        user=args.user,
        password=args.password,
        confirm=args.force
    ))