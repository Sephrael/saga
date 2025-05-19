#!/usr/bin/env python3
"""
Simple script to dump and inspect the knowledge graph from SAGA's database.
Run this from the SAGA directory: python inspect_kg.py
"""

import sqlite3
import json
from pathlib import Path

# Adjust this path if your database is elsewhere
DB_PATH = "novel_output/novel_data.db"

def inspect_knowledge_graph():
    if not Path(DB_PATH).exists():
        print(f"Database not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This lets us access columns by name
    
    try:
        cursor = conn.cursor()
        
        # Get basic stats
        cursor.execute("SELECT COUNT(*) as total FROM knowledge_graph")
        total_facts = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as provisional FROM knowledge_graph WHERE is_provisional = 1")
        provisional_facts = cursor.fetchone()['provisional']
        
        print(f"=== SAGA Knowledge Graph Summary ===")
        print(f"Total facts: {total_facts}")
        print(f"Provisional facts: {provisional_facts}")
        print(f"Reliable facts: {total_facts - provisional_facts}")
        print()
        
        # Get all triples, ordered by chapter and confidence
        cursor.execute("""
            SELECT subject, predicate, obj, chapter_added, confidence, is_provisional 
            FROM knowledge_graph 
            ORDER BY chapter_added, confidence DESC, subject
        """)
        
        current_chapter = None
        for row in cursor.fetchall():
            # Group by chapter for readability
            if row['chapter_added'] != current_chapter:
                current_chapter = row['chapter_added']
                chapter_label = f"Chapter {current_chapter}" if current_chapter > 0 else "Pre-novel/Setup"
                print(f"\n--- {chapter_label} ---")
            
            # Format the triple with metadata
            prov_marker = " [PROVISIONAL]" if row['is_provisional'] else ""
            conf_marker = f" (conf: {row['confidence']:.2f})" if row['confidence'] != 1.0 else ""
            
            print(f"  {row['subject']} -> {row['predicate']} -> {row['obj']}{conf_marker}{prov_marker}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def search_knowledge_graph(search_term=None):
    """Search for specific subjects, predicates, or objects"""
    if not search_term:
        search_term = input("Enter search term (or press Enter for all): ").strip()
    
    if not Path(DB_PATH).exists():
        print(f"Database not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        if search_term:
            # Search in subject, predicate, or object
            cursor.execute("""
                SELECT subject, predicate, obj, chapter_added, confidence, is_provisional 
                FROM knowledge_graph 
                WHERE subject LIKE ? OR predicate LIKE ? OR obj LIKE ?
                ORDER BY chapter_added, confidence DESC
            """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"))
            print(f"\n=== Search results for '{search_term}' ===")
        else:
            cursor.execute("""
                SELECT subject, predicate, obj, chapter_added, confidence, is_provisional 
                FROM knowledge_graph 
                ORDER BY chapter_added, confidence DESC
            """)
            print("\n=== All Knowledge Graph Facts ===")
        
        for row in cursor.fetchall():
            prov_marker = " [PROVISIONAL]" if row['is_provisional'] else ""
            conf_marker = f" (conf: {row['confidence']:.2f})" if row['confidence'] != 1.0 else ""
            chapter_label = f"Ch{row['chapter_added']}" if row['chapter_added'] > 0 else "Setup"
            
            print(f"  [{chapter_label}] {row['subject']} -> {row['predicate']} -> {row['obj']}{conf_marker}{prov_marker}")
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def export_to_json():
    """Export knowledge graph to JSON file"""
    if not Path(DB_PATH).exists():
        print(f"Database not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT subject, predicate, obj, chapter_added, confidence, is_provisional 
            FROM knowledge_graph 
            ORDER BY chapter_added, confidence DESC
        """)
        
        facts = []
        for row in cursor.fetchall():
            facts.append({
                'subject': row['subject'],
                'predicate': row['predicate'],
                'object': row['obj'],
                'chapter_added': row['chapter_added'],
                'confidence': row['confidence'],
                'is_provisional': bool(row['is_provisional'])
            })
        
        output_file = "knowledge_graph_export.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(facts, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {len(facts)} facts to {output_file}")
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("SAGA Knowledge Graph Inspector")
    print("1. View all facts (grouped by chapter)")
    print("2. Search for specific facts")
    print("3. Export to JSON")
    print("4. Just show summary")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    if choice == "1":
        inspect_knowledge_graph()
    elif choice == "2":
        search_knowledge_graph()
    elif choice == "3":
        export_to_json()
    elif choice == "4":
        # Just run the summary part
        import sqlite3
        from pathlib import Path
        
        if Path(DB_PATH).exists():
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM knowledge_graph")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) as prov FROM knowledge_graph WHERE is_provisional = 1")
            prov = cursor.fetchone()[0]
            conn.close()
            print(f"\nKnowledge Graph Summary:")
            print(f"  Total facts: {total}")
            print(f"  Provisional: {prov}")
            print(f"  Reliable: {total - prov}")
        else:
            print(f"Database not found at {DB_PATH}")
    else:
        print("Invalid choice")