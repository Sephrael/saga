# state_manager.py
import logging
import json
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import base64 # For encoding/decoding embeddings
import asyncio

# Neo4j specific imports
from neo4j import AsyncGraphDatabase, AsyncSession, AsyncManagedTransaction # type: ignore
from neo4j.exceptions import ServiceUnavailable, ClientError, Neo4jError # type: ignore

import config # For NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, EMBEDDING_DTYPE, etc.

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
        result = await tx.run(query, parameters)
        return await result.data() # Fetches all records

    async def _execute_read_query(self, query: str, parameters: Optional[Dict] = None):
        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")
        async with self.driver.session() as session:
            return await session.execute_read(self._execute_query_tx, query, parameters)

    async def _execute_write_query(self, query: str, parameters: Optional[Dict] = None):
        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")
        async with self.driver.session() as session:
            return await session.execute_write(self._execute_query_tx, query, parameters)
            
    async def execute_cypher_batch(self, cypher_statements_with_params: List[Tuple[str, Dict[str, Any]]]):
        """Executes a batch of Cypher statements in a single transaction."""
        if not cypher_statements_with_params:
            self.logger.info("execute_cypher_batch: No statements to execute.")
            return

        if self.driver is None: await self.connect()
        if self.driver is None: raise ConnectionError("Neo4j driver not initialized.")

        async with self.driver.session() as session: # type: AsyncSession
            tx = await session.begin_transaction() # type: AsyncManagedTransaction # FIX: await here
            try:
                for query, params in cypher_statements_with_params:
                    self.logger.debug(f"Batch Cypher: {query} with params {params}")
                    await tx.run(query, params)
                await tx.commit()
                self.logger.info(f"Successfully executed batch of {len(cypher_statements_with_params)} Cypher statements.")
            except Exception as e:
                self.logger.error(f"Error in Cypher batch execution: {e}. Rolling back.", exc_info=True)
                if tx.closed() is False: # Check if transaction is still open before attempting rollback
                    await tx.rollback()
                raise


    async def create_db_and_tables(self):
        self.logger.info("Creating/verifying Neo4j indexes and constraints...")
        # Core Node Types
        core_constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:NovelInfo) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chapter) REQUIRE c.number IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE", # General entities for KG
            "CREATE CONSTRAINT IF NOT EXISTS FOR (char:Character) REQUIRE char.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (we:WorldElement) REQUIRE we.id IS UNIQUE", # Using unique ID for world elements
            "CREATE CONSTRAINT IF NOT EXISTS FOR (wc:WorldContainer) REQUIRE wc.id IS UNIQUE", # Added for WorldContainer
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Trait) REQUIRE t.name IS UNIQUE", # Added for Trait
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pp:PlotPoint) REQUIRE pp.id IS UNIQUE", # Added for PlotPoint
            "CREATE CONSTRAINT IF NOT EXISTS FOR (vn:ValueNode) REQUIRE vn.value IS UNIQUE", # If ValueNodes should be unique by value
        ]
        # Indexes for faster lookups
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (pp:PlotPoint) ON (pp.sequence)",
            "CREATE INDEX IF NOT EXISTS FOR (s:StatusEvent) ON (s.chapter_updated)",
            "CREATE INDEX IF NOT EXISTS FOR (d:DevelopmentEvent) ON (d.chapter_updated)",
            "CREATE INDEX IF NOT EXISTS FOR (we:WorldElaborationEvent) ON (we.chapter_updated)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.chapter_added)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:DYNAMIC_REL]-() ON (r.type)", # Index on DYNAMIC_REL type
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:HAS_TRAIT]-() ON (r.name)", 
            "CREATE INDEX IF NOT EXISTS FOR (we:WorldElement) ON (we.category)",
            "CREATE INDEX IF NOT EXISTS FOR (we:WorldElement) ON (we.name)" 
        ]
        
        all_schema_ops = core_constraints + indexes
        for query in all_schema_ops:
            try:
                await self._execute_write_query(query)
            except Exception as e: 
                self.logger.warning(f"Failed to apply schema operation '{query}': {e} (This might be okay if it already exists in a slightly different form or due to concurrent setup).")
        self.logger.info("Neo4j indexes and constraints verification process complete.")

    def _serialize_embedding(self, embedding: np.ndarray) -> Tuple[str, str, str]:
        embedding_to_save = embedding.astype(config.EMBEDDING_DTYPE)
        if embedding_to_save.ndim == 0:
             embedding_to_save = embedding_to_save.reshape(1)
        return base64.b64encode(embedding_to_save.tobytes()).decode('utf-8'), \
               str(embedding_to_save.dtype), \
               json.dumps(list(embedding_to_save.shape))

    def _deserialize_embedding(self, b64_blob: str, dtype_str: str, shape_str: str) -> Optional[np.ndarray]:
        try:
            blob_bytes = base64.b64decode(b64_blob)
            shape = tuple(json.loads(shape_str))
            dtype = np.dtype(dtype_str)
            return np.frombuffer(blob_bytes, dtype=dtype).reshape(shape)
        except Exception as e:
            self.logger.error(f"Error deserializing embedding (shape: {shape_str}, dtype: {dtype_str}): {e}", exc_info=True)
            return None

    # --- Decomposed Plot Outline ---
          
    async def save_plot_outline(self, plot_data: Dict[str, Any]) -> bool:
        self.logger.info("Saving decomposed plot outline to Neo4j using MERGE...")
        if not plot_data:
            self.logger.warning("save_plot_outline: plot_data is empty. Nothing to save.")
            return False

        novel_id = config.MAIN_NOVEL_INFO_NODE_ID
        statements = []

        # Clear existing plot outline data first
        statements.append((f"MATCH (ni:NovelInfo {{id: '{novel_id}'}})-[r:HAS_PLOT_POINT]->(pp:PlotPoint) DETACH DELETE pp, r", {}))
        statements.append((f"MATCH (ni:NovelInfo {{id: '{novel_id}'}}) DETACH DELETE ni", {}))

        # Prepare NovelInfo properties
        novel_props_for_set = {k: v for k, v in plot_data.items() if not isinstance(v, (list, dict)) and v is not None}
        novel_props_for_set['id'] = novel_id # Ensure 'id' is in the properties for SET

        # MERGE NovelInfo node
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
                    # MERGE PlotPoint node
                    statements.append((
                        "MERGE (pp:PlotPoint {id: $id_val}) SET pp = $props",
                        {"id_val": pp_id, "props": pp_props_for_set}
                    ))
                    # MERGE relationship from NovelInfo to PlotPoint
                    statements.append((
                        f"""
                        MATCH (ni:NovelInfo {{id: '{novel_id}'}})
                        MATCH (pp:PlotPoint {{id: $pp_id_val}})
                        MERGE (ni)-[:HAS_PLOT_POINT]->(pp)
                        """,
                        {"pp_id_val": pp_id} # Use a different param name to avoid conflict if novel_id was $id_val
                    ))
                    if i > 0:
                        prev_pp_id = f"{novel_id}_pp_{i}"
                        # MERGE relationship from previous PlotPoint to current
                        statements.append((
                            f"""
                            MATCH (prev_pp:PlotPoint {{id: '{prev_pp_id}'}})
                            MATCH (curr_pp:PlotPoint {{id: $pp_id_val}})
                            MERGE (prev_pp)-[:NEXT_PLOT_POINT]->(curr_pp)
                            """,
                            {"pp_id_val": pp_id} # Use a different param name
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

        novel_info_query = f"MATCH (ni:NovelInfo {{id: '{novel_id}'}}) RETURN ni"
        result = await self._execute_read_query(novel_info_query)
        if not result or not result[0] or not result[0]['ni']:
            self.logger.warning(f"No NovelInfo node found with id '{novel_id}'. Returning empty plot outline.")
            return {}
        
        plot_data.update(result[0]['ni']) 
        plot_data_id = plot_data.pop('id', None) # Remove the 'id' property we set if it was NovelInfo node's id
        if plot_data_id != novel_id and plot_data_id is not None: # If a different 'id' was a property
            plot_data['id_prop_original'] = plot_data_id # Keep it under a different name

        plot_points_query = f"""
        MATCH (ni:NovelInfo {{id: '{novel_id}'}})-[:HAS_PLOT_POINT]->(pp:PlotPoint)
        RETURN pp.sequence AS sequence, pp.description AS description
        ORDER BY pp.sequence
        """
        pp_results = await self._execute_read_query(plot_points_query)
        plot_points_list = []
        if pp_results:
            for record in pp_results:
                plot_points_list.append(record['description'])
        plot_data['plot_points'] = plot_points_list
        
        self.logger.info("Successfully loaded and recomposed plot outline from Neo4j.")
        return plot_data

    # --- Decomposed Character Profiles ---
          
    async def save_character_profiles(self, profiles_data: Dict[str, Any]) -> bool:
        self.logger.info("Saving decomposed character profiles to Neo4j using MERGE...")
        if not profiles_data:
            self.logger.warning("save_character_profiles: profiles_data is empty. Nothing to save.")
            return False

        statements = []
        # Clear existing character-related data first
        statements.append(("MATCH (c:Character)-[r]->() DETACH DELETE r", {})) # Delete outgoing relationships
        statements.append(("MATCH ()-[r]->(c:Character) DETACH DELETE r", {})) # Delete incoming relationships
        statements.append(("MATCH (c:Character) DETACH DELETE c", {}))
        statements.append(("MATCH (t:Trait) DETACH DELETE t", {}))
        statements.append(("MATCH (dev:DevelopmentEvent) DETACH DELETE dev", {}))

        for char_name, profile in profiles_data.items():
            if not isinstance(profile, dict): continue

            char_props_for_set = {k: v for k, v in profile.items() if isinstance(v, (str, int, float, bool)) and v is not None}
            char_props_for_set['name'] = char_name # Ensure 'name' is in properties for SET

            # MERGE the Character node
            statements.append((
                "MERGE (c:Character {name: $char_name_val}) SET c = $props",
                {"char_name_val": char_name, "props": char_props_for_set}
            ))

            # Handle traits
            if isinstance(profile.get("traits"), list):
                for trait_str in profile["traits"]:
                    if isinstance(trait_str, str):
                        statements.append((
                            """
                            MATCH (c:Character {name: $char_name_val})
                            MERGE (t:Trait {name: $trait_name_val})
                            MERGE (c)-[:HAS_TRAIT]->(t)
                            """,
                            {"char_name_val": char_name, "trait_name_val": trait_str}
                        ))

            # Handle relationships (already uses MERGE for c2, which is good)
            if isinstance(profile.get("relationships"), dict):
                for target_char_name, rel_detail in profile["relationships"].items():
                    rel_type_str = "RELATED_TO"
                    # Default props for the relationship itself
                    rel_props_for_set = {"description": str(rel_detail), "chapter_added": config.KG_PREPOPULATION_CHAPTER_NUM, "is_provisional": False}
                    if isinstance(rel_detail, dict) and "type" in rel_detail:
                        rel_type_str = rel_detail.pop("type", rel_type_str).upper().replace(" ", "_")
                        rel_props_for_set.update(rel_detail) # Add any other props from rel_detail to relationship

                    statements.append((
                        """
                        MATCH (c1:Character {name: $char_name1_val})
                        MERGE (c2:Character {name: $char_name2_val})
                            ON CREATE SET c2.description = 'Placeholder description - created via relationship link from ' + $char_name1_val, c2.name = $char_name2_val
                        MERGE (c1)-[r:DYNAMIC_REL {type: $rel_type_val}]->(c2)
                        SET r += $rel_props_val, r.chapter_added = COALESCE($rel_props_val.chapter_added, r.chapter_added, $default_chap_add), r.is_provisional = COALESCE($rel_props_val.is_provisional, r.is_provisional, false)
                        """,
                        {
                            "char_name1_val": char_name,
                            "char_name2_val": target_char_name,
                            "rel_type_val": rel_type_str,
                            "rel_props_val": rel_props_for_set, # Pass the whole dict
                            "default_chap_add": config.KG_PREPOPULATION_CHAPTER_NUM
                        }
                    ))
            
            # Handle development events (these are new events, so CREATE is fine after character MERGE)
            for key, value_str in profile.items():
                if key.startswith("development_in_chapter_") and isinstance(value_str, str):
                    try:
                        chap_num_int = int(key.split("_")[-1])
                        dev_event_props = {
                            "summary": value_str,
                            "chapter": chap_num_int # Store as integer
                        }
                        statements.append((
                            """
                            MATCH (c:Character {name: $char_name_val})
                            CREATE (dev:DevelopmentEvent $props)
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

        char_query = "MATCH (c:Character) RETURN c"
        char_results = await self._execute_read_query(char_query)
        if not char_results:
            return {}

        for record in char_results:
            char_node = record['c']
            char_name = char_node.get('name')
            if not char_name:
                continue
            profiles_data[char_name] = dict(char_node)
            profiles_data[char_name].pop('name', None)  # Name is the key

            traits_query = "MATCH (c:Character {name: $char_name})-[:HAS_TRAIT]->(t:Trait) RETURN t.name AS trait_name"
            trait_results = await self._execute_read_query(traits_query, {"char_name": char_name})
            profiles_data[char_name]["traits"] = [tr['trait_name'] for tr in trait_results] if trait_results else []

            rels_query = """
            MATCH (c1:Character {name: $char_name})-[r:DYNAMIC_REL]->(c2:Character)
            RETURN c2.name AS target_name, r
            """
            rel_results = await self._execute_read_query(rels_query, {"char_name": char_name})
            relationships = {}
            if rel_results:
                for rel_rec in rel_results:
                    target_name = rel_rec['target_name']
                    rel_node_obj = rel_rec['r'] # This is the relationship object itself

                    if rel_node_obj:
                        rel_node_props = {k: v for k, v in rel_node_obj.items()}
                        # The 'type' is a property on DYNAMIC_REL, not its Cypher type.
                        # So it should already be in rel_node_props if it was set correctly.
                        # If 'type' is not in rel_node_props, it means it was missing during save or
                        # the relationship was created without it.
                        rel_type = rel_node_props.get("type", "RELATED_TO") # Default if somehow missing

                        # Logic to simplify representation if only description and type are present
                        if len(rel_node_props) == 2 and 'description' in rel_node_props and 'type' in rel_node_props:
                             relationships[target_name] = {"type": rel_type, "description": rel_node_props['description']}
                        elif len(rel_node_props) == 1 and 'type' in rel_node_props: # only type
                            relationships[target_name] = rel_type # Store just the type string
                        else: # multiple props, store the whole dict
                            relationships[target_name] = rel_node_props
                    else:
                        self.logger.warning(f"Encountered a null relationship object for target {target_name} from {char_name}")


            profiles_data[char_name]["relationships"] = relationships

            dev_query = """
            MATCH (c:Character {name: $char_name})-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)
            RETURN dev.summary AS summary, dev.chapter AS chapter
            """
            dev_results = await self._execute_read_query(dev_query, {"char_name": char_name})
            if dev_results:
                for dev_rec in dev_results:
                    profiles_data[char_name][f"development_in_chapter_{dev_rec['chapter']}"] = dev_rec['summary']

        self.logger.info(f"Successfully loaded and recomposed {len(profiles_data)} character profiles from Neo4j.")
        return profiles_data

    # --- Decomposed World Building ---
          
    async def save_world_building(self, world_data: Dict[str, Any]) -> bool:
        self.logger.info("Saving decomposed world building data to Neo4j using MERGE...")
        if not world_data:
            self.logger.warning("save_world_building: world_data is empty. Nothing to save.")
            return False

        statements = []
        # Clear existing world-related data first
        # Note: Be cautious with broad DETACH DELETE statements on a live/large graph.
        # For this system's current lifecycle (often full recreate/update), it's acceptable.
        statements.append(("MATCH (we:WorldElement)-[r]->() DETACH DELETE r", {}))
        statements.append(("MATCH ()-[r]->(we:WorldElement) DETACH DELETE r", {}))
        statements.append(("MATCH (we:WorldElement) DETACH DELETE we", {}))
        statements.append(("MATCH (wev:WorldElaborationEvent) DETACH DELETE wev", {}))
        statements.append((f"MATCH (wc:WorldContainer {{id: '{config.MAIN_WORLD_CONTAINER_NODE_ID}'}}) DETACH DELETE wc", {}))
        statements.append((f"MATCH (vn:ValueNode) DETACH DELETE vn", {})) # Clear ValueNodes

        for category_str, items_dict_value_from_world_data in world_data.items(): # Renamed for clarity
            # First, handle special top-level keys explicitly
            if category_str == "_overview_":
                if isinstance(items_dict_value_from_world_data, dict) and "description" in items_dict_value_from_world_data:
                    wc_id = config.MAIN_WORLD_CONTAINER_NODE_ID
                    desc_to_set = str(items_dict_value_from_world_data.get("description", "")) # Ensure string, default empty
                    wc_props = {
                        "id": wc_id,
                        "overview_description": desc_to_set
                    }
                    statements.append((
                        f"MERGE (wc:WorldContainer {{id: $id_val}}) SET wc = $props",
                        {"id_val": wc_id, "props": wc_props }
                    ))
                else:
                    # Log if _overview_ exists but is not in the expected format, or if it's missing a description
                    if category_str in world_data: # Check if _overview_ key actually exists
                        self.logger.warning(f"World data for '_overview_' category is not a dict with 'description' or is missing the description. Skipping save for overview details. Data: {items_dict_value_from_world_data}")
                    # If _overview_ key itself is missing, it's fine, nothing to save for it.
                continue # Move to the next category_str

            # Skip other known meta keys or if the value is not a dictionary (which would be an items category)
            if category_str in ["is_default", "source", "user_supplied_data"] or \
               not isinstance(items_dict_value_from_world_data, dict):
                if not isinstance(items_dict_value_from_world_data, dict):
                    self.logger.warning(f"Skipping world category '{category_str}' because its value is not a dictionary (type: {type(items_dict_value_from_world_data)}). Value: {items_dict_value_from_world_data}")
                else: # It's a dict, but it's one of the meta keys like "is_default"
                    self.logger.debug(f"Skipping meta world category '{category_str}'.")
                continue # Move to the next category_str

            # At this point, items_dict_value_from_world_data MUST be a dictionary representing a category of items
            items_category_dict = items_dict_value_from_world_data

            for item_name_str, details_dict in items_category_dict.items():
                # Skip items whose names indicate they are meta-properties within a category dict,
                # or if their details_dict is not actually a dictionary.
                if not isinstance(details_dict, dict) or \
                   item_name_str.startswith(("_", "source_quality_chapter_", "category_updated_in_chapter_")):
                    if not isinstance(details_dict, dict):
                        self.logger.warning(f"Skipping item '{item_name_str}' in world category '{category_str}' because its details_dict is not a dictionary (type: {type(details_dict)}). Value: {details_dict}")
                    # else: (it's a meta-key starting with _ or other recognized prefixes for items)
                    # self.logger.debug(f"Skipping meta-item '{item_name_str}' in category '{category_str}'.")
                    continue # Skip this specific item and go to the next item in the category

                we_id_str = f"{category_str}_{item_name_str}".replace(" ", "_").replace("'", "").lower()
                # Prepare properties for the WorldElement node itself
                item_props_for_set = {k: v for k, v in details_dict.items() if isinstance(v, (str, int, float, bool)) and v is not None}
                item_props_for_set['id'] = we_id_str
                item_props_for_set['name'] = item_name_str
                item_props_for_set['category'] = category_str
                # Ensure created_chapter is present, defaulting if necessary
                item_props_for_set['created_chapter'] = details_dict.get('created_chapter', details_dict.get('added_in_chapter_0', config.KG_PREPOPULATION_CHAPTER_NUM))


                # MERGE WorldElement node
                statements.append((
                    "MERGE (we:WorldElement {id: $id_val}) SET we = $props",
                    {"id_val": we_id_str, "props": item_props_for_set}
                ))

                # Handle list properties (goals, rules, key_elements) by MERGING ValueNodes and relationships
                for list_prop_key_str in ["goals", "rules", "key_elements"]: 
                    list_value = details_dict.get(list_prop_key_str)
                    if isinstance(list_value, list):
                        for val_item_from_list in list_value: 
                            if isinstance(val_item_from_list, str):
                                rel_name_internal_str = f"HAS_{list_prop_key_str.upper().rstrip('S')}"
                                if list_prop_key_str == "key_elements": 
                                    rel_name_internal_str = "HAS_KEY_ELEMENT"
                                
                                statements.append((
                                    f"""
                                    MATCH (we:WorldElement {{id: $we_id_val}})
                                    MERGE (v:ValueNode {{value: $val_item_value, type: '{list_prop_key_str}'}})
                                    MERGE (we)-[:{rel_name_internal_str}]->(v)
                                    """,
                                    {"we_id_val": we_id_str, "val_item_value": val_item_from_list}
                                ))

                # Handle elaboration events (CREATE is fine here as they are new per chapter, after WorldElement MERGE)
                for key_str, value_val in details_dict.items():
                    if key_str.startswith("elaboration_in_chapter_") and isinstance(value_val, str):
                        try:
                            chap_num_val = int(key_str.split("_")[-1])
                            elab_props = {
                                "summary": value_val,
                                "chapter": chap_num_val
                            }
                            statements.append((
                                """
                                MATCH (we:WorldElement {{id: $we_id_val}})
                                CREATE (we_elab:WorldElaborationEvent $props)
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
                self.logger.info("No statements generated for saving world building data (world_data might be empty or only contain meta-keys).")
            return True
        except Exception as e:
            self.logger.error(f"Error saving decomposed world building data with MERGE: {e}", exc_info=True)
            return False

    async def get_world_building(self) -> Dict[str, Any]:
        self.logger.info("Loading decomposed world building data from Neo4j...")
        world_data: Dict[str, Any] = {}

        overview_query = f"MATCH (wc:WorldContainer {{id: '{config.MAIN_WORLD_CONTAINER_NODE_ID}'}}) RETURN wc.overview_description AS desc"
        overview_res = await self._execute_read_query(overview_query)
        if overview_res and overview_res[0] and overview_res[0].get('desc'):
            world_data["_overview_"] = {"description": overview_res[0]['desc']}
        else:
            world_data["_overview_"] = {}


        we_query = "MATCH (we:WorldElement) RETURN we"
        we_results = await self._execute_read_query(we_query)
        if not we_results: 
            # Ensure all standard categories exist even if empty
            for cat_key in ["locations", "society", "systems", "lore", "history", "factions"]:
                if cat_key not in world_data: world_data[cat_key] = {}
            return world_data

        for record in we_results:
            we_node = record['we']
            category = we_node.get('category')
            item_name = we_node.get('name')
            we_id = we_node.get('id')

            if not category or not item_name or not we_id: continue

            if category not in world_data:
                world_data[category] = {}
            
            item_detail = dict(we_node) 
            item_detail.pop('id', None); item_detail.pop('name', None); item_detail.pop('category', None)
            item_detail.pop('created_chapter', None) # Internal tracking
            
            # Corrected loop for list properties
            for list_prop_key in ["goals", "rules", "key_elements"]:
                rel_name_query = f"HAS_{list_prop_key.upper().rstrip('S')}"
                if list_prop_key == "key_elements":
                    rel_name_query = "HAS_KEY_ELEMENT"

                list_values_query = f"""
                MATCH (we:WorldElement {{id: $we_id}})-[:{rel_name_query}]->(v:ValueNode {{type: '{list_prop_key}'}})
                RETURN v.value AS item_value
                """
                list_val_res = await self._execute_read_query(list_values_query, {"we_id": we_id})
                if list_val_res:
                    item_detail[list_prop_key] = [res_item['item_value'] for res_item in list_val_res]
                else: 
                    item_detail[list_prop_key] = []


            elab_query = """
            MATCH (we:WorldElement {id: $we_id})-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent)
            RETURN elab.summary AS summary, elab.chapter AS chapter
            """
            elab_results = await self._execute_read_query(elab_query, {"we_id": we_id})
            if elab_results:
                for elab_rec in elab_results:
                    item_detail[f"elaboration_in_chapter_{elab_rec['chapter']}"] = elab_rec['summary']
            
            world_data[category][item_name] = item_detail
        
        # Ensure all standard categories exist even if empty after processing
        for cat_key in ["locations", "society", "systems", "lore", "history", "factions"]:
            if cat_key not in world_data: world_data[cat_key] = {}

        self.logger.info(f"Successfully loaded and recomposed world building data from Neo4j.")
        return world_data

    async def async_load_chapter_count(self) -> int:
        query = "MATCH (c:Chapter) RETURN count(c) AS chapter_count"
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

        embedding_b64, embedding_dtype, embedding_shape = None, None, None
        if embedding_array is not None and isinstance(embedding_array, np.ndarray) and embedding_array.size > 0:
            embedding_b64, embedding_dtype, embedding_shape = self._serialize_embedding(embedding_array)

        query = """
        MERGE (c:Chapter {number: $chapter_number})
        SET c.text = $text,
            c.raw_llm_output = $raw_llm_output,
            c.summary = $summary,
            c.is_provisional = $is_provisional,
            c.embedding_b64 = $embedding_b64,
            c.embedding_dtype = $embedding_dtype,
            c.embedding_shape = $embedding_shape 
        """ 
        parameters = {
            "chapter_number": chapter_number,
            "text": text,
            "raw_llm_output": raw_llm_output,
            "summary": summary if summary is not None else "",
            "is_provisional": is_provisional,
            "embedding_b64": embedding_b64,
            "embedding_dtype": embedding_dtype,
            "embedding_shape": embedding_shape
        }
        try:
            await self._execute_write_query(query, parameters)
            self.logger.info(f"Neo4j: Successfully saved chapter data (including embedding) for chapter {chapter_number}.")
        except Exception as e:
            self.logger.error(f"Neo4j: Error saving chapter data for chapter {chapter_number}: {e}", exc_info=True)

    async def async_get_chapter_data_from_db(self, chapter_number: int) -> Optional[Dict[str, Any]]:
        if chapter_number <= 0: return None
        query = """
        MATCH (c:Chapter {number: $chapter_number})
        RETURN c.text AS text, c.raw_llm_output AS raw_llm_output, c.summary AS summary, c.is_provisional AS is_provisional
        """
        try:
            result = await self._execute_read_query(query, {"chapter_number": chapter_number})
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
        query = """
        MATCH (c:Chapter {number: $chapter_number})
        WHERE c.embedding_b64 IS NOT NULL AND c.embedding_dtype IS NOT NULL AND c.embedding_shape IS NOT NULL
        RETURN c.embedding_b64 AS embedding_b64, c.embedding_dtype AS dtype, c.embedding_shape AS shape
        """
        try:
            result = await self._execute_read_query(query, {"chapter_number": chapter_number})
            if result and result[0] and result[0].get("embedding_b64"): # Check if embedding_b64 is not None
                return self._deserialize_embedding(result[0]["embedding_b64"], result[0]["dtype"], result[0]["shape"])
            self.logger.debug(f"Neo4j: No embedding found directly on chapter node {chapter_number}.")
            return None
        except Exception as e:
            self.logger.error(f"Neo4j: Error getting embedding for {chapter_number}: {e}", exc_info=True)
            return None

    async def async_get_all_past_embeddings(self, current_chapter_number: int) -> List[Tuple[int, np.ndarray]]:
        embeddings_list: List[Tuple[int, np.ndarray]] = []
        query = """
        MATCH (c:Chapter)
        WHERE c.number < $current_chapter_number AND c.number > 0
          AND c.embedding_b64 IS NOT NULL AND c.embedding_dtype IS NOT NULL AND c.embedding_shape IS NOT NULL
        RETURN c.number AS chapter_number, c.embedding_b64 AS embedding_b64, c.embedding_dtype AS dtype, c.embedding_shape AS shape
        ORDER BY c.number DESC
        """
        try:
            results = await self._execute_read_query(query, {"current_chapter_number": current_chapter_number})
            if results:
                for record in results:
                    if record.get("embedding_b64"): # Ensure embedding data exists
                        deserialized_emb = self._deserialize_embedding(record["embedding_b64"], record["dtype"], record["shape"])
                        if deserialized_emb is not None:
                            embeddings_list.append((record["chapter_number"], deserialized_emb))
            self.logger.info(f"Neo4j: Retrieved {len(embeddings_list)} past embeddings.")
            return embeddings_list
        except Exception as e:
            self.logger.error(f"Neo4j: Error getting all past embeddings: {e}", exc_info=True)
            return []
        
    async def async_add_kg_triple(self, subject: str, predicate: str, obj_val: str, chapter_added: int, confidence: float = 1.0, is_provisional: bool = False):
        subj_s, pred_s, obj_s = subject.strip(), predicate.strip(), obj_val.strip()
        if not all([subj_s, pred_s, obj_s]) or chapter_added < config.KG_PREPOPULATION_CHAPTER_NUM:
            self.logger.warning(f"Neo4j: Invalid KG triple for add: S='{subj_s}', P='{pred_s}', O='{obj_s}', Chap={chapter_added}")
            return

        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:DYNAMIC_REL {type: $predicate, chapter_added: $chapter_added, is_provisional: $is_provisional}]->(o)
        SET r.confidence = $confidence 
        RETURN s.name, r.type, o.name
        """
        parameters = {
            "subject": subj_s,
            "predicate": pred_s, 
            "object": obj_s,
            "chapter_added": chapter_added,
            "confidence": confidence,
            "is_provisional": is_provisional
        }
        try:
            await self._execute_write_query(query, parameters)
            self.logger.debug(f"Neo4j: Added/Updated KG triple for Ch {chapter_added}: ({subj_s}, {pred_s}, {obj_s}). Prov: {is_provisional}")
        except Exception as e:
            self.logger.error(f"Neo4j: Error adding KG triple: ({subj_s}, {pred_s}, {obj_s}). Error: {e}", exc_info=True)

    async def async_query_kg(self, subject: Optional[str] = None, predicate: Optional[str] = None, obj_val: Optional[str] = None, chapter_limit: Optional[int] = None, include_provisional: bool = True, limit_results: Optional[int] = None) -> List[Dict[str, Any]]:
        conditions = []
        parameters = {}
        
        match_clause = "MATCH (s:Entity)-[r:DYNAMIC_REL]->(o:Entity)"

        if subject is not None:
            conditions.append("s.name = $subject")
            parameters["subject"] = subject.strip()
        if predicate is not None:
            conditions.append("r.type = $predicate") 
            parameters["predicate"] = predicate.strip()
        if obj_val is not None:
            conditions.append("o.name = $object")
            parameters["object"] = obj_val.strip()
        if chapter_limit is not None:
            conditions.append("r.chapter_added <= $chapter_limit")
            parameters["chapter_limit"] = chapter_limit
        if not include_provisional:
            conditions.append("r.is_provisional = FALSE")
        
        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
        
        return_clause = " RETURN s.name AS subject, r.type AS predicate, o.name AS object, r.chapter_added AS chapter_added, r.confidence AS confidence, r.is_provisional AS is_provisional"
        order_clause = " ORDER BY r.chapter_added DESC, r.confidence DESC"
        limit_clause = ""
        if limit_results is not None and limit_results > 0:
            limit_clause = f" LIMIT {limit_results}"

        full_query = match_clause + where_clause + return_clause + order_clause + limit_clause

        try:
            results = await self._execute_read_query(full_query, parameters)
            triples_list: List[Dict[str, Any]] = []
            if results:
                for record in results:
                    triples_list.append(dict(record)) 
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
        if results and results[0]:
            value = str(results[0]["object"]) 
            self.logger.debug(f"Neo4j: Found most recent value for ('{subject}', '{predicate}'): '{value}' from Ch {results[0]['chapter_added']}")
            return value
        self.logger.debug(f"Neo4j: No value found for ({subject}, {predicate}) up to Ch {chapter_limit}, provisional={include_provisional}")
        return None

    # --- Getters for prompt_data_getters (NEW/REVISED) ---
    async def get_character_info_for_snippet(self, char_name: str, chapter_limit: int) -> Optional[Dict[str, Any]]:
        """Gets description, latest status, and latest development note for a character from Neo4j."""
        query = """
        MATCH (c:Character {name: $char_name})
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent)
        WHERE dev.chapter <= $chapter_limit OR dev IS NULL
        WITH c, dev ORDER BY dev.chapter DESC
        // Collect all development summaries up to the chapter_limit
        WITH c, COLLECT(CASE WHEN dev IS NOT NULL THEN dev.summary ELSE NULL END)[0] AS recent_dev_summary_val
        // Determine provisional status based on any DYNAMIC_REL marked provisional for this character as subject or object
        // or if the character node itself has a provisional marker (if schema supports that)
        OPTIONAL MATCH (c)-[rel_s:DYNAMIC_REL {is_provisional: true}]->() WHERE rel_s.chapter_added <= $chapter_limit
        OPTIONAL MATCH ()-[rel_o:DYNAMIC_REL {is_provisional: true}]->(c) WHERE rel_o.chapter_added <= $chapter_limit
        RETURN c.description AS description, 
               c.status AS current_status, // Assuming status is a direct property
               recent_dev_summary_val AS most_recent_development_note,
               (COUNT(rel_s) > 0 OR COUNT(rel_o) > 0) AS is_provisional_overall 
               // Simplified: if any relevant KG fact is provisional, mark char as having some provisional info
        LIMIT 1
        """
        params = {"char_name": char_name, "chapter_limit": chapter_limit}
        try:
            result = await self._execute_read_query(query, params)
            if result and result[0]:
                record = result[0]
                return {
                    "description": record.get("description"),
                    "current_status": record.get("current_status"),
                    "most_recent_development_note": record.get("most_recent_development_note") if record.get("most_recent_development_note") else "N/A",
                    "is_provisional_overall": record.get("is_provisional_overall", False)
                }
            self.logger.debug(f"No detailed snippet info found for character '{char_name}' in Neo4j up to chapter {chapter_limit}.")
        except Exception as e:
            self.logger.error(f"Error fetching character info for snippet ({char_name}) from Neo4j: {e}", exc_info=True)
        return None

    async def get_world_elements_for_snippet(self, category: str, chapter_limit: int, item_limit: int) -> List[Dict[str, Any]]:
        """Gets key world elements for a category from Neo4j."""
        query = """
        MATCH (we:WorldElement {category: $category})
        // OPTIONAL MATCH (we)-[:ELABORATED_IN_CHAPTER]->(elab:WorldElaborationEvent) // If needed
        // WHERE (elab.chapter <= $chapter_limit OR elab IS NULL)
        WITH we //, elab ORDER BY elab.chapter DESC
        LIMIT $item_limit // Apply limit early if possible based on category elements
        // Determine provisional status based on DYNAMIC_RELs involving this WorldElement
        OPTIONAL MATCH (we)-[rel_s:DYNAMIC_REL {is_provisional: true}]->() WHERE rel_s.chapter_added <= $chapter_limit
        OPTIONAL MATCH ()-[rel_o:DYNAMIC_REL {is_provisional: true}]->(we) WHERE rel_o.chapter_added <= $chapter_limit
        RETURN we.name AS name, 
               we.description AS description, 
               // HEAD(COLLECT(elab.summary)) as last_elaboration, // If elaborations are used for snippet
               (COUNT(rel_s) > 0 OR COUNT(rel_o) > 0) AS is_provisional
        ORDER BY we.name 
        """ # Removed last_elaboration for simplicity, description snippet is primary
        params = {"category": category, "chapter_limit": chapter_limit, "item_limit": item_limit}
        items = []
        try:
            results = await self._execute_read_query(query, params)
            if results:
                for record in results:
                    items.append({
                        "name": record.get("name"),
                        "description_snippet": (record.get("description") or "")[:50] + "...", # Snippet from description
                        "is_provisional": record.get("is_provisional", False)
                    })
        except Exception as e:
            self.logger.error(f"Error fetching world elements for snippet (category {category}) from Neo4j: {e}", exc_info=True)
        return items

state_manager = state_managerSingleton()