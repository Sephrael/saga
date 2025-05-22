#!/usr/bin/env python3
"""
Simple script to dump and inspect the knowledge graph from SAGA's Neo4j database.
Run this from the SAGA directory: python inspect_kg.py
"""

import asyncio
from neo4j import AsyncGraphDatabase
import json
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MAIN_NOVEL_INFO_NODE_ID

driver = None

async def connect_to_neo4j():
    global driver
    if driver is None:
        try:
            driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            await driver.verify_connectivity()
            print(f"Connected to Neo4j at {NEO4J_URI}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            driver = None
            raise
    return driver

async def close_neo4j():
    global driver
    if driver:
        await driver.close()
        driver = None
        print("Neo4j connection closed.")

async def inspect_knowledge_graph():
    db_driver = await connect_to_neo4j()
    if not db_driver: return

    async with db_driver.session() as session:
        # Get basic stats
        total_facts_result = await session.run("MATCH ()-[r:DYNAMIC_REL]->() RETURN count(r) as total")
        total_facts_record = await total_facts_result.single()
        total_facts = total_facts_record["total"] if total_facts_record else 0
        
        provisional_facts_result = await session.run("MATCH ()-[r:DYNAMIC_REL {is_provisional: true}]->() RETURN count(r) as provisional")
        provisional_facts_record = await provisional_facts_result.single()
        provisional_facts = provisional_facts_record["provisional"] if provisional_facts_record else 0
        
        print(f"=== SAGA Knowledge Graph Summary (DYNAMIC_RELs) ===")
        print(f"Total DYNAMIC_REL facts: {total_facts}")
        print(f"Provisional DYNAMIC_REL facts: {provisional_facts}")
        print(f"Reliable DYNAMIC_REL facts: {total_facts - provisional_facts}")
        
        # Count other key nodes
        chapter_count_res = await session.run("MATCH (c:Chapter) RETURN count(c) as count")
        char_count_res = await session.run("MATCH (c:Character) RETURN count(c) as count")
        world_el_count_res = await session.run("MATCH (w:WorldElement) RETURN count(w) as count")
        print(f"Total Chapter nodes: {(await chapter_count_res.single())['count']}")
        print(f"Total Character nodes: {(await char_count_res.single())['count']}")
        print(f"Total WorldElement nodes: {(await world_el_count_res.single())['count']}")
        print()
        
        # Get all DYNAMIC_REL triples, ordered by chapter and confidence
        query = """
            MATCH (s:Entity)-[r:DYNAMIC_REL]->(o:Entity)
            RETURN s.name AS subject, r.type AS predicate, o.name AS obj, 
                   r.chapter_added AS chapter_added, r.confidence AS confidence, r.is_provisional AS is_provisional
            ORDER BY r.chapter_added, r.confidence DESC, s.name
            LIMIT 200 // Limit for inspectability
        """
        result = await session.run(query)
        
        current_chapter = -999 # Init with a value that won't match chapter_added
        async for record in result:
            if record['chapter_added'] != current_chapter:
                current_chapter = record['chapter_added']
                chapter_label = f"Chapter {current_chapter}" if current_chapter > 0 else \
                                f"Pre-novel/Setup (Ch {config.KG_PREPOPULATION_CHAPTER_NUM})" if current_chapter == config.KG_PREPOPULATION_CHAPTER_NUM else \
                                f"Chapter {current_chapter} (Unknown)"
                print(f"\n--- {chapter_label} ---")
            
            prov_marker = " [PROVISIONAL]" if record['is_provisional'] else ""
            conf_val = record['confidence']
            conf_marker = f" (conf: {conf_val:.2f})" if conf_val is not None and conf_val != 1.0 else "" # Check for None
            
            print(f"  {record['subject']} -> {record['predicate']} -> {record['obj']}{conf_marker}{prov_marker}")

async def search_knowledge_graph(search_term=None):
    db_driver = await connect_to_neo4j()
    if not db_driver: return

    if not search_term:
        search_term = input("Enter search term (or press Enter for all DYNAMIC_RELs): ").strip()
    
    async with db_driver.session() as session:
        if search_term:
            query = """
                MATCH (s:Entity)-[r:DYNAMIC_REL]->(o:Entity)
                WHERE s.name CONTAINS $term OR r.type CONTAINS $term OR o.name CONTAINS $term
                RETURN s.name AS subject, r.type AS predicate, o.name AS obj, 
                       r.chapter_added AS chapter_added, r.confidence AS confidence, r.is_provisional AS is_provisional
                ORDER BY r.chapter_added, r.confidence DESC
                LIMIT 100
            """
            params = {"term": search_term}
            print(f"\n=== Search results for '{search_term}' (DYNAMIC_RELs) ===")
        else:
            query = """
                MATCH (s:Entity)-[r:DYNAMIC_REL]->(o:Entity)
                RETURN s.name AS subject, r.type AS predicate, o.name AS obj, 
                       r.chapter_added AS chapter_added, r.confidence AS confidence, r.is_provisional AS is_provisional
                ORDER BY r.chapter_added, r.confidence DESC
                LIMIT 100
            """
            params = {}
            print("\n=== All DYNAMIC_REL Knowledge Graph Facts (Limit 100) ===")
        
        result = await session.run(query, params)
        async for record in result:
            prov_marker = " [PROVISIONAL]" if record['is_provisional'] else ""
            conf_val = record['confidence']
            conf_marker = f" (conf: {conf_val:.2f})" if conf_val is not None and conf_val != 1.0 else ""
            chapter_label = f"Ch{record['chapter_added']}" if record['chapter_added'] > 0 else \
                            f"Setup{config.KG_PREPOPULATION_CHAPTER_NUM}" if record['chapter_added'] == config.KG_PREPOPULATION_CHAPTER_NUM else \
                            f"Ch{record['chapter_added']}"
            
            print(f"  [{chapter_label}] {record['subject']} -> {record['predicate']} -> {record['obj']}{conf_marker}{prov_marker}")

async def export_to_json():
    db_driver = await connect_to_neo4j()
    if not db_driver: return

    async with db_driver.session() as session:
        # Export DYNAMIC_REL facts
        dynamic_rel_query = """
            MATCH (s:Entity)-[r:DYNAMIC_REL]->(o:Entity)
            RETURN s.name AS subject, r.type AS predicate, o.name AS object, 
                   r.chapter_added AS chapter_added, r.confidence AS confidence, r.is_provisional AS is_provisional
            ORDER BY r.chapter_added, r.confidence DESC
        """
        dynamic_rel_result = await session.run(dynamic_rel_query)
        facts = [dict(record) async for record in dynamic_rel_result] # Convert records to dicts

        # Optionally, export other structures like Character nodes, PlotPoints, etc.
        # Example: Character export
        # character_query = "MATCH (c:Character) RETURN c"
        # character_result = await session.run(character_query)
        # characters = [dict(record['c']) async for record in character_result]
        # output_data = {"dynamic_rels": facts, "characters": characters}
        
        output_data = {"dynamic_rels": facts} # Simplified for now

        output_file = "neo4j_knowledge_graph_export.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str) # Added default=str for Neo4j types
        
        print(f"Exported {len(facts)} DYNAMIC_REL facts to {output_file}")

async def main_inspector():
    try:
        print("SAGA Neo4j Knowledge Graph Inspector")
        print("1. View DYNAMIC_REL facts (grouped by chapter, limit 200)")
        print("2. Search DYNAMIC_REL facts (limit 100)")
        print("3. Export DYNAMIC_REL facts to JSON")
        print("4. Just show summary counts")
        
        choice = input("\nChoose option (1-4): ").strip()
        
        if choice == "1":
            await inspect_knowledge_graph()
        elif choice == "2":
            await search_knowledge_graph()
        elif choice == "3":
            await export_to_json()
        elif choice == "4":
            db_driver = await connect_to_neo4j()
            if not db_driver: return
            async with db_driver.session() as session:
                total_facts_result = await session.run("MATCH ()-[r:DYNAMIC_REL]->() RETURN count(r) as total")
                total_facts_record = await total_facts_result.single()
                total = total_facts_record["total"] if total_facts_record else 0
                
                provisional_facts_result = await session.run("MATCH ()-[r:DYNAMIC_REL {is_provisional: true}]->() RETURN count(r) as provisional")
                provisional_facts_record = await provisional_facts_result.single()
                prov = provisional_facts_record["provisional"] if provisional_facts_record else 0
                
                print(f"\nKnowledge Graph Summary (DYNAMIC_RELs):")
                print(f"  Total DYNAMIC_REL facts: {total}")
                print(f"  Provisional DYNAMIC_REL facts: {prov}")
                print(f"  Reliable DYNAMIC_REL facts: {total - prov}")
        else:
            print("Invalid choice")
    finally:
        await close_neo4j()

if __name__ == "__main__":
    try:
        asyncio.run(main_inspector())
    except KeyboardInterrupt:
        print("\nInspector interrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")