# reset_neo4j.py

from neo4j import GraphDatabase
import argparse
import time

def reset_neo4j_database(uri="bolt://localhost:7687", user="neo4j", password="saga_password", confirm=False):
    """
    Resets a Neo4j database by removing all nodes and relationships.
    
    Args:
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        confirm: If False, will ask for confirmation before deleting
    
    Returns:
        True if reset was successful, False otherwise
    """
    if not confirm:
        response = input("⚠️ WARNING: This will delete ALL data in the Neo4j database. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            return False
    
    driver = None
    try:
        print("Connecting to Neo4j database...")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            print(f"Current database has {node_count} nodes.")
        
        # Perform the reset - delete all relationships and nodes
        print("Resetting database...")
        start_time = time.time()
        
        with driver.session() as session:
            # First delete all relationships
            result = session.run("MATCH ()-[r]-() DELETE r RETURN count(r) as deleted_relationships")
            rel_count = result.single()["deleted_relationships"]
            
            # Then delete all nodes
            result = session.run("MATCH (n) DELETE n RETURN count(n) as deleted_nodes")
            nodes_deleted = result.single()["deleted_nodes"]
        
        elapsed_time = time.time() - start_time
        print(f"✅ Database reset complete! Deleted {rel_count} relationships and {nodes_deleted} nodes in {elapsed_time:.2f} seconds.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        return False
    
    finally:
        if driver:
            driver.close()
            print("Connection closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset a Neo4j database by removing all nodes and relationships")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="saga_password", help="Neo4j password")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    reset_neo4j_database(
        uri=args.uri,
        user=args.user,
        password=args.password,
        confirm=args.force
    )