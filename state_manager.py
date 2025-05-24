# state_manager.py
import logging
import json
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np
# import base64 # No longer needed for embeddings if storing as LIST<FLOAT>
import asyncio

# Neo4j specific imports
from neo4j import AsyncGraphDatabase, AsyncSession, AsyncManagedTransaction # type: ignore
from neo4j.exceptions import ServiceUnavailable, ClientError, Neo4jError # type: ignore

import config 

logger = logging.getLogger(__name__)

class state_managerSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(state_managerSingleton, cls).__new__(cls)
            cls._instance._initialized_flag = False
        return cls._instance

    def __init__(self):
        if self._initialized_flag:
            return
        
        self.logger = logging.getLogger(__name__)
        self.driver: Optional[AsyncGraphDatabase] = None
        self._initialized_flag = True
        self.logger.info("Neo4j state_managerSingleton initialized. Call connect() to establish connection.")

    async def connect(self):
        if self.driver is None:
            try:
                self.driver = AsyncGraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
                await self.driver.verify_connectivity()
                self.logger.info(f"Successfully connected to Neo4j at {config.NEO4J_URI}")
            except ServiceUnavailable as e:
                self.logger.critical(f"Neo4j connection failed: {e}. Ensure the Neo4j database is running and accessible.")
                self.driver = None
                raise
            except Exception as e:
                self.logger.critical(f"Unexpected error during Neo4j connection: {e}", exc_info=True)
                self.driver = None
                raise

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None
            self.logger.info("Neo4j driver closed.")

    async def _execute_query_tx(self, tx: AsyncManagedTransaction, query: str, parameters: Optional[Dict] = None):
        self.logger.debug(f"Executing Cypher query: {query} with params: {parameters}")
        result = await tx.run(query, parameters)
        return await result.data() 

    async def _execute_read_query(self, query: str, parameters: Optional[Dict] = None):
        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")
        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
            return await session.execute_read(self._execute_query_tx, query, parameters)

    async def _execute_write_query(self, query: str, parameters: Optional[Dict] = None):
        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")
        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: ignore
            return await session.execute_write(self._execute_query_tx, query, parameters)
            
    async def execute_cypher_batch(self, cypher_statements_with_params: List[Tuple[str, Dict[str, Any]]]):
        """Executes a batch of Cypher statements in a single transaction."""
        if not cypher_statements_with_params:
            self.logger.info("execute_cypher_batch: No statements to execute.")
            return

        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")

        async with self.driver.session(database=config.NEO4J_DATABASE) as session: # type: AsyncSession # type: ignore
            tx: AsyncManagedTransaction = await session.begin_transaction() 
            try:
                for query, params in cypher_statements_with_params:
                    self.logger.debug(f"Batch Cypher: {query} with params {params}")
                    await tx.run(query, params)
                await tx.commit()
                self.logger.info(f"Successfully executed batch of {len(cypher_statements_with_params)} Cypher statements.")
            except Exception as e:
                self.logger.error(f"Error in Cypher batch execution: {e}. Rolling back.", exc_info=True)
                if tx.closed() is False: 
                    await tx.rollback()
                raise


    async def create_db_and_tables(self): # Renamed to reflect Neo4j schema setup
        self.logger.info("Creating/verifying Neo4j indexes and constraints, including vector index...")
        
        core_constraints = [
            "CREATE CONSTRAINT novelInfo_id_unique IF NOT EXISTS FOR (n:NovelInfo) REQUIRE n.id IS UNIQUE",
            f"CREATE CONSTRAINT chapter_number_unique IF NOT EXISTS FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) REQUIRE c.number IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT character_name_unique IF NOT EXISTS FOR (char:Character) REQUIRE char.name IS UNIQUE",
            "CREATE CONSTRAINT worldElement_id_unique IF NOT EXISTS FOR (we:WorldElement) REQUIRE we.id IS UNIQUE",
            "CREATE CONSTRAINT worldContainer_id_unique IF NOT EXISTS FOR (wc:WorldContainer) REQUIRE wc.id IS UNIQUE",
            "CREATE CONSTRAINT trait_name_unique IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT plotPoint_id_unique IF NOT EXISTS FOR (pp:PlotPoint) REQUIRE pp.id IS UNIQUE",
            "CREATE CONSTRAINT valueNode_value_type_unique IF NOT EXISTS FOR (vn:ValueNode) REQUIRE (vn.value, vn.type) IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX plotPoint_sequence IF NOT EXISTS FOR (pp:PlotPoint) ON (pp.sequence)",
            "CREATE INDEX statusEvent_chapter_updated IF NOT EXISTS FOR (s:StatusEvent) ON (s.chapter_updated)", 
            "CREATE INDEX developmentEvent_chapter_updated IF NOT EXISTS FOR (d:DevelopmentEvent) ON (d.chapter_updated)",
            "CREATE INDEX worldElaborationEvent_chapter_updated IF NOT EXISTS FOR (we:WorldElaborationEvent) ON (we.chapter_updated)",
            "CREATE INDEX dynamicRel_chapter_added IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.chapter_added)",
            "CREATE INDEX dynamicRel_type IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.type)",
            "CREATE INDEX worldElement_category IF NOT EXISTS FOR (we:WorldElement) ON (we.category)",
            "CREATE INDEX worldElement_name IF NOT EXISTS FOR (we:WorldElement) ON (we.name)",
            f"CREATE INDEX chapter_is_provisional IF NOT EXISTS FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) ON (c.is_provisional)", 
            "CREATE INDEX dynamicRel_is_provisional IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.is_provisional)",
        ]

        # Vector Index Creation
        # Note: Neo4j versions might have slightly different syntax. This is for 5.x.
        # For Aura or specific versions, check documentation.
        vector_index_query = f"""
        CREATE VECTOR INDEX {config.NEO4J_VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:{config.NEO4J_VECTOR_NODE_LABEL}) ON (c.{config.NEO4J_VECTOR_PROPERTY_NAME})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: {config.NEO4J_VECTOR_DIMENSIONS},
            `vector.similarity_function`: '{config.NEO4J_VECTOR_SIMILARITY_FUNCTION}'
        }}}}
        """
        
        all_schema_ops = core_constraints + indexes + [vector_index_query]
        for query in all_schema_ops:
            try:
                await self._execute_write_query(query)
                if "VECTOR INDEX" in query:
                    self.logger.info(f"Successfully created/verified vector index: {config.NEO4J_VECTOR_INDEX_NAME}")
            except Exception as e: 
                self.logger.warning(f"Failed to apply schema operation '{query}': {e} (This might be okay if it already exists or due to concurrent setup).")
        self.logger.info("Neo4j schema (indexes, constraints, vector index) verification process complete.")

    def _embedding_to_list(self, embedding: Optional[np.ndarray]) -> Optional[List[float]]:
        if embedding is None:
            return None
        if not isinstance(embedding, np.ndarray):
            self.logger.warning(f"Attempting to convert non-numpy array to list for Neo4j: {type(embedding)}")
            if hasattr(embedding, 'tolist'):
                return embedding.tolist() # type: ignore
            return None
        return embedding.astype(np.float32).tolist()

    def _list_to_embedding(self, embedding_list: Optional[List[Union[float, int]]]) -> Optional[np.ndarray]:
        if embedding_list is None:
            return None
        try:
            return np.array(embedding_list, dtype=config.EMBEDDING_DTYPE)
        except Exception as e:
            self.logger.error(f"Error converting list to numpy embedding: {e}", exc_info=True)
            return None

    async def save_plot_outline(self, plot_data: Dict[str, Any]) -> bool:
        self.logger.info("Saving decomposed plot outline to Neo4j using MERGE...")
        if not plot_data:
            self.logger.warning("save_plot_outline: plot_data is empty. Nothing to save.")
            return False

        novel_id = config.MAIN_NOVEL_INFO_NODE_ID
        statements = []
        statements.append((
            """
            MATCH (ni:NovelInfo {id: $novel_id_param})
            OPTIONAL MATCH (ni)-[r_has_pp:HAS_PLOT_POINT]->(pp:PlotPoint)
            DETACH DELETE pp, r_has_pp
            """,
            {"novel_id_param": novel_id}
        ))
        statements.append((
            "MATCH (ni:NovelInfo {id: $novel_id_param}) DETACH DELETE ni",
            {"novel_id_param": novel_id}
        ))
        novel_props_for_set = {k: v for k, v in plot_data.items() if not isinstance(v, (list, dict)) and v is not None}
        novel_props_for_set['id'] = novel_id 
        statements.append((
            "MERGE (ni:NovelInfo {id: $id_val}) SET ni = $props",
            {"id_val": novel_id, "props": novel_props_for_set}
        ))
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
                    if i > 0:
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
            await self.execute_cypher_batch(statements)
            self.logger.info("Successfully saved decomposed plot outline to Neo4j using MERGE.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving decomposed plot outline with MERGE: {e}", exc_info=True)
            return False

    async def get_plot_outline(self) -> Dict[str, Any]:
        self.logger.info("Loading decomposed plot outline from Neo4j...")
        novel_id = config.MAIN_NOVEL_INFO_NODE_ID
        plot_data: Dict[str, Any] = {}
        novel_info_query = "MATCH (ni:NovelInfo {id: $novel_id_param}) RETURN ni"
        result = await self._execute_read_query(novel_info_query, {"novel_id_param": novel_id})
        if not result or not result[0] or not result[0].get('ni'):
            self.logger.warning(f"No NovelInfo node found with id '{novel_id}'. Returning empty plot outline.")
            return {}
        plot_data.update(result[0]['ni']) 
        plot_data.pop('id', None) 
        plot_points_query = """
        MATCH (ni:NovelInfo {id: $novel_id_param})-[:HAS_PLOT_POINT]->(pp:PlotPoint)
        RETURN pp.description AS description
        ORDER BY pp.sequence ASC
        """
        pp_results = await self._execute_read_query(plot_points_query, {"novel_id_param": novel_id})
        plot_data['plot_points'] = [record['description'] for record in pp_results] if pp_results else []
        self.logger.info("Successfully loaded and recomposed plot outline from Neo4j.")
        return plot_data
          
    async def save_character_profiles(self, profiles_data: Dict[str, Any]) -> bool:
        self.logger.info("Saving decomposed character profiles to Neo4j using MERGE...")
        if not profiles_data:
            self.logger.warning("save_character_profiles: profiles_data is empty. Nothing to save.")
            return False
        statements = []
        statements.append(("MATCH (c:Character)-[r:HAS_TRAIT]->() DELETE r", {}))
        statements.append(("MATCH (c:Character)-[r:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent) DELETE r, dev", {}))
        statements.append(("MATCH (c1:Character)-[r:DYNAMIC_REL]-(c2:Character) DELETE r", {}))
        statements.append(("MATCH (c:Character) REMOVE c:Character", {}))
        statements.append(("MATCH (t:Trait) DETACH DELETE t", {}))
        for char_name, profile in profiles_data.items():
            if not isinstance(profile, dict): continue
            char_props_for_set = {k: v for k, v in profile.items() if isinstance(v, (str, int, float, bool)) and v is not None}
            character_node_query = """
            MERGE (c:Entity {name: $char_name_val}) 
            SET c:Character                         
            SET c += $props                         
            """
            statements.append((
                character_node_query, 
                {"char_name_val": char_name, "props": char_props_for_set}
            ))
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
            if isinstance(profile.get("relationships"), dict):
                for target_char_name, rel_detail in profile["relationships"].items():
                    rel_type_str = "RELATED_TO" 
                    rel_props_for_set = {"description": str(rel_detail)}
                    if isinstance(rel_detail, dict):
                        rel_type_str = str(rel_detail.get("type", rel_type_str)).upper().replace(" ", "_")
                        rel_props_for_set = {k:v for k,v in rel_detail.items() if isinstance(v, (str, int, float, bool))}
                        rel_props_for_set.setdefault("description", f"{rel_type_str} {target_char_name}") 
                    elif isinstance(rel_detail, str):
                        rel_type_str = rel_detail.upper().replace(" ", "_") 
                        rel_props_for_set = {"description": rel_detail}
                    rel_props_for_set.setdefault("chapter_added", profile.get(f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}", config.KG_PREPOPULATION_CHAPTER_NUM))
                    rel_props_for_set.setdefault("is_provisional", profile.get(f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}") == "provisional_from_unrevised_draft" if f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}" in profile else False)
                    statements.append((
                        """
                        MATCH (c1:Character:Entity {name: $char_name1_val})
                        MERGE (c2:Entity {name: $char_name2_val}) 
                            ON CREATE SET c2:Character, c2.description = 'Placeholder desc - created via rel from ' + $char_name1_val 
                            ON MATCH SET c2:Character 
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
            for key, value_str in profile.items():
                if key.startswith("development_in_chapter_") and isinstance(value_str, str):
                    try:
                        chap_num_int = int(key.split("_")[-1])
                        dev_event_props = {
                            "summary": value_str,
                            "chapter_updated": chap_num_int 
                        }
                        provisional_dev = profile.get(f"source_quality_chapter_{chap_num_int}") == "provisional_from_unrevised_draft"
                        if provisional_dev:
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
                        self.logger.warning(f"Could not parse chapter number from development key: {key}")
        try:
            await self.execute_cypher_batch(statements)
            self.logger.info("Successfully saved decomposed character profiles to Neo4j using MERGE.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving decomposed character profiles with MERGE: {e}", exc_info=True)
            return False

    async def get_character_profiles(self) -> Dict[str, Any]:
        self.logger.info("Loading decomposed character profiles from Neo4j...")
        profiles_data: Dict[str, Any] = {}
        char_query = "MATCH (c:Character:Entity) RETURN c" 
        char_results = await self._execute_read_query(char_query)
        if not char_results:
            return {}
        for record in char_results:
            char_node = record['c']
            char_name = char_node.get('name')
            if not char_name:
                continue
            profile = dict(char_node)
            profile.pop('name', None) 
            traits_query = "MATCH (:Character:Entity {name: $char_name})-[:HAS_TRAIT]->(t:Trait) RETURN t.name AS trait_name"
            trait_results = await self._execute_read_query(traits_query, {"char_name": char_name})
            profile["traits"] = [tr['trait_name'] for tr in trait_results] if trait_results else []
            rels_query = """
            MATCH (:Character:Entity {name: $char_name})-[r:DYNAMIC_REL]->(target:Character:Entity) 
            RETURN target.name AS target_name, r.type AS relationship_type, properties(r) AS rel_props
            """
            rel_results = await self._execute_read_query(rels_query, {"char_name": char_name})
            relationships = {}
            if rel_results:
                for rel_rec in rel_results:
                    target_name = rel_rec['target_name']
                    rel_type = rel_rec['relationship_type']
                    rel_props = rel_rec['rel_props']
                    relationships[target_name] = {**rel_props, "type": rel_type} if rel_props else {"type": rel_type}
            profile["relationships"] = relationships
            dev_query = """
            MATCH (:Character:Entity {name: $char_name})-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)
            RETURN dev.summary AS summary, dev.chapter_updated AS chapter, dev.is_provisional AS is_provisional
            """
            dev_results = await self._execute_read_query(dev_query, {"char_name": char_name})
            if dev_results:
                for dev_rec in dev_results:
                    dev_key = f"development_in_chapter_{dev_rec['chapter']}"
                    profile[dev_key] = dev_rec['summary']
                    if dev_rec.get('is_provisional'): 
                        profile[f"source_quality_chapter_{dev_rec['chapter']}"] = "provisional_from_unrevised_draft"
            profiles_data[char_name] = profile
        self.logger.info(f"Successfully loaded and recomposed {len(profiles_data)} character profiles from Neo4j.")
        return profiles_data
          
    async def save_world_building(self, world_data: Dict[str, Any]) -> bool:
        self.logger.info("Saving decomposed world building data to Neo4j using MERGE...")
        if not world_data:
            self.logger.warning("save_world_building: world_data is empty. Nothing to save.")
            return False
        statements = []
        statements.append(("MATCH (we:WorldElement) OPTIONAL MATCH (we)-[r]-() DETACH DELETE we, r", {}))
        statements.append(("MATCH (wev:WorldElaborationEvent) DETACH DELETE wev", {}))
        statements.append(("MATCH (wc:WorldContainer {id: $wc_id_param}) DETACH DELETE wc", {"wc_id_param": config.MAIN_WORLD_CONTAINER_NODE_ID}))
        statements.append(("MATCH (vn:ValueNode) DETACH DELETE vn", {})) 
        for category_str, items_dict_value_from_world_data in world_data.items(): 
            if category_str == "_overview_":
                if isinstance(items_dict_value_from_world_data, dict) and "description" in items_dict_value_from_world_data:
                    wc_id = config.MAIN_WORLD_CONTAINER_NODE_ID
                    desc_to_set = str(items_dict_value_from_world_data.get("description", "")) 
                    wc_props = {
                        "id": wc_id,
                        "overview_description": desc_to_set
                    }
                    if items_dict_value_from_world_data.get(f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}") == "provisional_from_unrevised_draft":
                         wc_props["is_provisional"] = True
                    statements.append((
                        "MERGE (wc:WorldContainer {id: $id_val}) SET wc = $props",
                        {"id_val": wc_id, "props": wc_props }
                    ))
                continue 
            if category_str in ["is_default", "source", "user_supplied_data"] or \
               not isinstance(items_dict_value_from_world_data, dict):
                continue 
            items_category_dict = items_dict_value_from_world_data
            for item_name_str, details_dict in items_category_dict.items():
                if not isinstance(details_dict, dict) or \
                   item_name_str.startswith(("_", "source_quality_chapter_", "category_updated_in_chapter_")):
                    continue 
                we_id_str = f"{category_str}_{item_name_str}".replace(" ", "_").replace("'", "").lower()
                item_props_for_set = {k: v for k, v in details_dict.items() if isinstance(v, (str, int, float, bool)) and v is not None}
                item_props_for_set['id'] = we_id_str
                item_props_for_set['name'] = item_name_str
                item_props_for_set['category'] = category_str
                created_chap_num = config.KG_PREPOPULATION_CHAPTER_NUM
                is_item_provisional = False
                added_key = next((k for k in details_dict if k.startswith("added_in_chapter_")), None)
                if added_key:
                    try: created_chap_num = int(added_key.split("_")[-1])
                    except ValueError: pass
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
                for list_prop_key_str in ["goals", "rules", "key_elements", "traits"]: 
                    list_value = details_dict.get(list_prop_key_str)
                    if isinstance(list_value, list):
                        for val_item_from_list in list_value: 
                            if isinstance(val_item_from_list, str):
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
                for key_str, value_val in details_dict.items():
                    if key_str.startswith("elaboration_in_chapter_") and isinstance(value_val, str):
                        try:
                            chap_num_val = int(key_str.split("_")[-1])
                            elab_props = {
                                "summary": value_val,
                                "chapter_updated": chap_num_val 
                            }
                            provisional_elab = details_dict.get(f"source_quality_chapter_{chap_num_val}") == "provisional_from_unrevised_draft"
                            if provisional_elab:
                                elab_props["is_provisional"] = True
                            statements.append((
                                """
                                MATCH (we:WorldElement {id: $we_id_val})
                                CREATE (we_elab:WorldElaborationEvent)
                                SET we_elab = $props
                                CREATE (we)-[:ELABORATED_IN_CHAPTER]->(we_elab)
                                """,
                                {"we_id_val": we_id_str, "props": elab_props}
                            ))
                        except ValueError:
                            self.logger.warning(f"Could not parse chapter number from world elaboration key: {key_str}")
        try:
            if statements: 
                await self.execute_cypher_batch(statements)
                self.logger.info("Successfully saved decomposed world building data to Neo4j using MERGE.")
            else:
                self.logger.info("No statements generated for saving world building data.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving decomposed world building data with MERGE: {e}", exc_info=True)
            return False

    async def get_world_building(self) -> Dict[str, Any]:
        self.logger.info("Loading decomposed world building data from Neo4j...")
        world_data: Dict[str, Any] = {"_overview_": {}}
        overview_query = "MATCH (wc:WorldContainer {id: $wc_id_param}) RETURN wc.overview_description AS desc, wc.is_provisional AS is_provisional"
        overview_res = await self._execute_read_query(overview_query, {"wc_id_param": config.MAIN_WORLD_CONTAINER_NODE_ID})
        if overview_res and overview_res[0] and overview_res[0].get('desc') is not None: 
            world_data["_overview_"]["description"] = overview_res[0]['desc']
            if overview_res[0].get('is_provisional'):
                 world_data["_overview_"][f"source_quality_chapter_{config.KG_PREPOPULATION_CHAPTER_NUM}"] = "provisional_from_unrevised_draft"
        we_query = "MATCH (we:WorldElement) RETURN we"
        we_results = await self._execute_read_query(we_query)
        standard_categories = ["locations", "society", "systems", "lore", "history", "factions"]
        for cat_key in standard_categories:
            world_data.setdefault(cat_key, {}) 
        if not we_results: 
            return world_data
        for record in we_results:
            we_node = record['we']
            category = we_node.get('category')
            item_name = we_node.get('name')
            we_id = we_node.get('id')
            if not category or not item_name or not we_id: continue
            if category not in world_data: world_data[category] = {} 
            item_detail = dict(we_node) 
            item_detail.pop('id', None); item_detail.pop('name', None); item_detail.pop('category', None)
            created_chapter_num = item_detail.pop('created_chapter', config.KG_PREPOPULATION_CHAPTER_NUM)
            item_detail[f"added_in_chapter_{created_chapter_num}"] = True 
            if item_detail.pop('is_provisional', False):
                item_detail[f"source_quality_chapter_{created_chapter_num}"] = "provisional_from_unrevised_draft"
            for list_prop_key in ["goals", "rules", "key_elements", "traits"]:
                rel_name_query = f"HAS_{list_prop_key.upper().rstrip('S')}"
                if list_prop_key == "key_elements": rel_name_query = "HAS_KEY_ELEMENT"
                elif list_prop_key == "traits": rel_name_query = "HAS_TRAIT_ASPECT"
                list_values_query = """
                MATCH (we_node:WorldElement {id: $we_id_param})-[:{rel_name_query}]->(v:ValueNode {{type: $value_node_type_param}})
                RETURN v.value AS item_value
                """.replace("we_node:", ":WorldElement") # Corrected alias usage
                formatted_list_values_query = list_values_query.format(rel_name_query=rel_name_query)
                list_val_res = await self._execute_read_query(formatted_list_values_query, {"we_id_param": we_id, "value_node_type_param": list_prop_key})
                item_detail[list_prop_key] = [res_item['item_value'] for res_item in list_val_res] if list_val_res else []
            elab_query = """
            MATCH (:WorldElement {id: $we_id_param})-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent)
            RETURN elab.summary AS summary, elab.chapter_updated AS chapter, elab.is_provisional AS is_provisional
            """
            elab_results = await self._execute_read_query(elab_query, {"we_id_param": we_id})
            if elab_results:
                for elab_rec in elab_results:
                    elab_key = f"elaboration_in_chapter_{elab_rec['chapter']}"
                    item_detail[elab_key] = elab_rec['summary']
                    if elab_rec.get('is_provisional'):
                        item_detail[f"source_quality_chapter_{elab_rec['chapter']}"] = "provisional_from_unrevised_draft"
            world_data[category][item_name] = item_detail
        self.logger.info(f"Successfully loaded and recomposed world building data from Neo4j.")
        return world_data

    async def async_load_chapter_count(self) -> int:
        query = f"MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL}) RETURN count(c) AS chapter_count"
        try:
            result = await self._execute_read_query(query)
            count = result[0]["chapter_count"] if result and result[0] else 0
            self.logger.info(f"Neo4j loaded chapter count: {count}")
            return count
        except Exception as e:
            self.logger.error(f"Failed to load chapter count from Neo4j: {e}", exc_info=True)
            return 0

    async def async_save_chapter_data(self, chapter_number: int, text: str, raw_llm_output: str, summary: Optional[str], embedding_array: Optional[np.ndarray], is_provisional: bool = False):
        if chapter_number <= 0:
            self.logger.error(f"Neo4j: Cannot save chapter data for invalid chapter_number: {chapter_number}.")
            return

        embedding_list = self._embedding_to_list(embedding_array)

        query = f"""
        MERGE (c:{config.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
        SET c.text = $text_param,
            c.raw_llm_output = $raw_llm_output_param,
            c.summary = $summary_param,
            c.is_provisional = $is_provisional_param,
            c.{config.NEO4J_VECTOR_PROPERTY_NAME} = $embedding_vector_param,
            c.last_updated = timestamp()
        """ 
        parameters = {
            "chapter_number_param": chapter_number,
            "text_param": text,
            "raw_llm_output_param": raw_llm_output,
            "summary_param": summary if summary is not None else "",
            "is_provisional_param": is_provisional,
            "embedding_vector_param": embedding_list, # Store as list of floats
        }
        try:
            await self._execute_write_query(query, parameters)
            self.logger.info(f"Neo4j: Successfully saved chapter data (embedding as list) for chapter {chapter_number}.")
        except Exception as e:
            self.logger.error(f"Neo4j: Error saving chapter data for chapter {chapter_number}: {e}", exc_info=True)

    async def async_get_chapter_data_from_db(self, chapter_number: int) -> Optional[Dict[str, Any]]:
        if chapter_number <= 0: return None
        query = f"""
        MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
        RETURN c.text AS text, c.raw_llm_output AS raw_llm_output, c.summary AS summary, c.is_provisional AS is_provisional
        """
        try:
            result = await self._execute_read_query(query, {"chapter_number_param": chapter_number})
            if result and result[0]:
                self.logger.debug(f"Neo4j: Data found for chapter {chapter_number}.")
                return {
                    "text": result[0].get("text"),
                    "summary": result[0].get("summary"),
                    "is_provisional": result[0].get("is_provisional", False), 
                    "raw_llm_output": result[0].get("raw_llm_output") 
                }
            self.logger.debug(f"Neo4j: No data found for chapter {chapter_number}.")
            return None
        except Exception as e:
            self.logger.error(f"Neo4j: Error getting chapter data for {chapter_number}: {e}", exc_info=True)
            return None

    async def async_get_embedding_from_db(self, chapter_number: int) -> Optional[np.ndarray]:
        if chapter_number <= 0: return None
        query = f"""
        MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL} {{number: $chapter_number_param}})
        WHERE c.{config.NEO4J_VECTOR_PROPERTY_NAME} IS NOT NULL
        RETURN c.{config.NEO4J_VECTOR_PROPERTY_NAME} AS embedding_vector
        """
        try:
            result = await self._execute_read_query(query, {"chapter_number_param": chapter_number})
            if result and result[0] and result[0].get("embedding_vector"): 
                embedding_list = result[0]["embedding_vector"]
                return self._list_to_embedding(embedding_list)
            self.logger.debug(f"Neo4j: No embedding vector found on chapter node {chapter_number}.")
            return None
        except Exception as e:
            self.logger.error(f"Neo4j: Error getting embedding for {chapter_number}: {e}", exc_info=True)
            return None

    async def async_find_similar_chapters(self, query_embedding: np.ndarray, limit: int, current_chapter_to_exclude: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Finds chapters with embeddings similar to the query_embedding using Neo4j's vector index.
        Returns a list of dicts, each containing 'chapter_number', 'summary', 'text', 'is_provisional', and 'score'.
        """
        if query_embedding is None or query_embedding.size == 0:
            self.logger.warning("Neo4j: async_find_similar_chapters called with empty query_embedding.")
            return []

        query_embedding_list = self._embedding_to_list(query_embedding)
        if query_embedding_list is None:
            self.logger.error("Neo4j: Failed to convert query_embedding to list for similarity search.")
            return []
            
        # Exclude the current chapter itself from similarity results if provided
        exclude_clause = ""
        params = {"queryVector": query_embedding_list, "limit": limit}
        if current_chapter_to_exclude is not None:
            exclude_clause = "WHERE c.number <> $current_chapter_to_exclude_param "
            params["current_chapter_to_exclude_param"] = current_chapter_to_exclude

        # Ensure the chapter node exists and has the embedding property before calling yield
        # The WHERE c[config.NEO4J_VECTOR_PROPERTY_NAME] IS NOT NULL is good practice
        # though the index itself implies this for nodes it covers.
        similarity_query = f"""
        CALL db.index.vector.queryNodes($index_name_param, $limit_param, $queryVector_param)
        YIELD node AS c, score
        {exclude_clause} 
        RETURN c.number AS chapter_number, 
               c.summary AS summary, 
               c.text AS text, 
               c.is_provisional AS is_provisional, 
               score
        ORDER BY score DESC
        """
        # Update params for the specific call
        final_params = {
            "index_name_param": config.NEO4J_VECTOR_INDEX_NAME,
            "limit_param": limit + (1 if current_chapter_to_exclude is not None else 0), # Fetch a bit more to filter out current
            "queryVector_param": query_embedding_list
        }
        if current_chapter_to_exclude is not None:
            final_params["current_chapter_to_exclude_param"] = current_chapter_to_exclude


        similar_chapters_data: List[Dict[str, Any]] = []
        try:
            results = await self._execute_read_query(similarity_query, final_params)
            if results:
                for record in results:
                    # Double check exclusion here if the Cypher WHERE clause wasn't sufficient due to index interaction
                    if current_chapter_to_exclude is not None and record.get("chapter_number") == current_chapter_to_exclude:
                        continue
                    if len(similar_chapters_data) < limit:
                        similar_chapters_data.append({
                            "chapter_number": record.get("chapter_number"),
                            "summary": record.get("summary"),
                            "text": record.get("text"), # Potentially large, consider if only summary is needed
                            "is_provisional": record.get("is_provisional", False),
                            "score": record.get("score")
                        })
                    else:
                        break # Reached limit
            self.logger.info(f"Neo4j: Vector search found {len(similar_chapters_data)} similar chapters (limit {limit}).")
        except Exception as e:
            self.logger.error(f"Neo4j: Error during vector similarity search: {e}", exc_info=True)
        
        return similar_chapters_data

    async def async_get_all_past_embeddings(self, current_chapter_number: int) -> List[Tuple[int, np.ndarray]]:
        """
        DEPRECATED in favor of async_find_similar_chapters.
        Kept for now if any other part of the system relies on fetching all embeddings.
        Consider removing if semantic context is the only consumer.
        """
        self.logger.warning("async_get_all_past_embeddings is deprecated. Use async_find_similar_chapters for semantic context.")
        embeddings_list: List[Tuple[int, np.ndarray]] = []
        query = f"""
        MATCH (c:{config.NEO4J_VECTOR_NODE_LABEL})
        WHERE c.number < $current_chapter_number_param AND c.number > 0 
          AND c.{config.NEO4J_VECTOR_PROPERTY_NAME} IS NOT NULL
        RETURN c.number AS chapter_number, c.{config.NEO4J_VECTOR_PROPERTY_NAME} AS embedding_vector
        ORDER BY c.number DESC
        """
        try:
            results = await self._execute_read_query(query, {"current_chapter_number_param": current_chapter_number})
            if results:
                for record in results:
                    if record.get("embedding_vector"): 
                        deserialized_emb = self._list_to_embedding(record["embedding_vector"])
                        if deserialized_emb is not None:
                            embeddings_list.append((record["chapter_number"], deserialized_emb))
            self.logger.info(f"Neo4j (Deprecated Call): Retrieved {len(embeddings_list)} past embeddings for context before chapter {current_chapter_number}.")
            return embeddings_list
        except Exception as e:
            self.logger.error(f"Neo4j (Deprecated Call): Error getting all past embeddings: {e}", exc_info=True)
            return []
        
    async def async_add_kg_triple(self, subject: str, predicate: str, obj_val: str, chapter_added: int, confidence: float = 1.0, is_provisional: bool = False):
        subj_s, pred_s, obj_s = subject.strip(), predicate.strip(), obj_val.strip()
        if not all([subj_s, pred_s, obj_s]) or chapter_added < config.KG_PREPOPULATION_CHAPTER_NUM: 
            self.logger.warning(f"Neo4j: Invalid KG triple for add: S='{subj_s}', P='{pred_s}', O='{obj_s}', Chap={chapter_added}")
            return
        query = """
        MERGE (s:Entity {name: $subject_param})
        MERGE (o:Entity {name: $object_param})
        WITH s, o, $predicate_param AS pred_param, $chapter_added_param AS chap_param, 
             $is_provisional_param AS prov_param, $confidence_param AS conf_param 
        OPTIONAL MATCH (s)-[existing_r:DYNAMIC_REL {type: pred_param, chapter_added: chap_param}]->(o)
        FOREACH (r IN CASE WHEN existing_r IS NOT NULL THEN [existing_r] ELSE [] END |
          SET r.is_provisional = prov_param, 
              r.confidence = conf_param, 
              r.last_updated = timestamp()
        )
        FOREACH (ignoreMe IN CASE WHEN existing_r IS NULL THEN [1] ELSE [] END |
          CREATE (s)-[new_r:DYNAMIC_REL {
            type: pred_param, 
            chapter_added: chap_param, 
            is_provisional: prov_param, 
            confidence: conf_param,
            created_at: timestamp(),
            last_updated: timestamp()
          }]->(o)
        )
        """
        parameters = {
            "subject_param": subj_s,
            "predicate_param": pred_s, 
            "object_param": obj_s,
            "chapter_added_param": chapter_added,
            "confidence_param": confidence,
            "is_provisional_param": is_provisional
        }
        try:
            await self._execute_write_query(query, parameters)
            self.logger.debug(f"Neo4j: Added/Updated KG triple for Ch {chapter_added}: ({subj_s}, {pred_s}, {obj_s}). Prov: {is_provisional}, Conf: {confidence}")
        except Exception as e:
            self.logger.error(f"Neo4j: Error adding KG triple: ({subj_s}, {pred_s}, {obj_s}). Error: {e}", exc_info=True)

    async def async_query_kg(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj_val: Optional[str] = None, chapter_limit: Optional[int] = None, include_provisional: bool = True, limit_results: Optional[int] = None) -> List[Dict[str, Any]]:
        conditions = []
        parameters = {}
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
            conditions.append("r.is_provisional = FALSE")
        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        return_clause = """
        RETURN s.name AS subject, r.type AS predicate, o.name AS object, 
               r.chapter_added AS chapter_added, r.confidence AS confidence, r.is_provisional AS is_provisional
        """
        order_clause = " ORDER BY r.chapter_added DESC, r.confidence DESC"
        limit_clause = ""
        if limit_results is not None and limit_results > 0:
            limit_clause = f" LIMIT {int(limit_results)}" 
        full_query = match_clause + where_clause + return_clause + order_clause + limit_clause
        try:
            results = await self._execute_read_query(full_query, parameters)
            triples_list: List[Dict[str, Any]] = [dict(record) for record in results] if results else []
            self.logger.debug(f"Neo4j: KG query returned {len(triples_list)} results for query: {full_query} with params {parameters}")
            return triples_list
        except Exception as e:
            self.logger.error(f"Neo4j: Error querying KG. Query: {full_query}, Params: {parameters}, Error: {e}", exc_info=True)
            return []

    async def async_get_most_recent_value(self, subject: str, predicate: str, chapter_limit: Optional[int] = None, include_provisional: bool = False) -> Optional[str]:
        if not subject.strip() or not predicate.strip():
            self.logger.warning(f"Neo4j: get_most_recent_value: empty subject or predicate. S='{subject}', P='{predicate}'")
            return None
        results = await self.async_query_kg(
            subject=subject, 
            predicate=predicate, 
            chapter_limit=chapter_limit, 
            include_provisional=include_provisional,
            limit_results=1 
        )
        if results and results[0] and 'object' in results[0]:
            value = str(results[0]["object"]) 
            self.logger.debug(f"Neo4j: Found most recent value for ('{subject}', '{predicate}'): '{value}' from Ch {results[0].get('chapter_added','N/A')}")
            return value
        self.logger.debug(f"Neo4j: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}, provisional={include_provisional}")
        return None

    async def get_character_info_for_snippet(self, char_name: str, chapter_limit: int) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (c:Character:Entity {name: $char_name_param})
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)
        WHERE dev.chapter_updated <= $chapter_limit_param
        WITH c, dev ORDER BY dev.chapter_updated DESC
        WITH c, HEAD(COLLECT(dev)) AS latest_dev_event 
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(prov_dev:DevelopmentEvent)
        WHERE prov_dev.chapter_updated <= $chapter_limit_param AND prov_dev.is_provisional = TRUE
        OPTIONAL MATCH (c)-[prov_rel:DYNAMIC_REL]-(:Entity)
        WHERE prov_rel.chapter_added <= $chapter_limit_param AND prov_rel.is_provisional = TRUE
        
        RETURN c.description AS description,
               c.status AS current_status, 
               latest_dev_event.summary AS most_recent_development_note,
               (prov_dev IS NOT NULL OR prov_rel IS NOT NULL OR c.is_provisional = TRUE) AS is_provisional_overall
        LIMIT 1
        """ 
        params = {"char_name_param": char_name, "chapter_limit_param": chapter_limit}
        try:
            result = await self._execute_read_query(query, params)
            if result and result[0]:
                record = result[0]
                dev_note = record.get("most_recent_development_note") if record.get("most_recent_development_note") is not None else "N/A"
                return {
                    "description": record.get("description"),
                    "current_status": record.get("current_status"),
                    "most_recent_development_note": dev_note,
                    "is_provisional_overall": record.get("is_provisional_overall", False)
                }
            self.logger.debug(f"No detailed snippet info found for character '{char_name}' in Neo4j up to chapter {chapter_limit}.")
        except Exception as e:
            self.logger.error(f"Error fetching character info for snippet ({char_name}) from Neo4j: {e}", exc_info=True)
        return None

    async def get_world_elements_for_snippet(self, category: str, chapter_limit: int, item_limit: int) -> List[Dict[str, Any]]:
        query = """
        MATCH (we:WorldElement {category: $category_param})
        OPTIONAL MATCH (we)-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent)
        WHERE elab.chapter_updated <= $chapter_limit_param AND elab.is_provisional = TRUE
        
        WITH we, COLLECT(DISTINCT elab) AS provisional_elaborations 
        WITH we, (we.is_provisional = TRUE OR size(provisional_elaborations) > 0) AS is_item_provisional
        
        RETURN we.name AS name,
               we.description AS description, 
               is_item_provisional AS is_provisional
        ORDER BY we.name ASC 
        LIMIT $item_limit_param
        """
        params = {"category_param": category, "chapter_limit_param": chapter_limit, "item_limit_param": item_limit}
        items = []
        try:
            results = await self._execute_read_query(query, params)
            if results:
                for record in results:
                    desc = record.get("description") or ""
                    items.append({
                        "name": record.get("name"),
                        "description_snippet": (desc[:50].strip() + "..." if len(desc) > 50 else desc.strip()), 
                        "is_provisional": record.get("is_provisional", False)
                    })
        except Exception as e:
            self.logger.error(f"Error fetching world elements for snippet (category {category}) from Neo4j: {e}", exc_info=True)
        return items

state_manager = state_managerSingleton()