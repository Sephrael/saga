# data_access/character_queries.py
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import config
from core_db.base_db_manager import neo4j_manager
from kg_constants import (
    KG_IS_PROVISIONAL,
    KG_NODE_CHAPTER_UPDATED,
)
from kg_maintainer.models import CharacterProfile

logger = logging.getLogger(__name__)


async def sync_full_state_from_object_to_db(profiles_data: Dict[str, Any]) -> bool:
    logger.info("Synchronizing character profiles to Neo4j (non-destructive)...")

    novel_id_param = config.MAIN_NOVEL_INFO_NODE_ID
    statements: List[Tuple[str, Dict[str, Any]]] = []

    all_input_char_names: Set[str] = set(profiles_data.keys())

    try:
        existing_char_records = await neo4j_manager.execute_read_query(
            "MATCH (c:Character:Entity) RETURN c.name AS name"
        )
        existing_db_char_names: Set[str] = {
            record["name"]
            for record in existing_char_records
            if record and record["name"]
        }
    except Exception as e:
        logger.error(
            f"Failed to retrieve existing character names from DB: {e}", exc_info=True
        )
        return False

    chars_to_delete = existing_db_char_names - all_input_char_names
    if chars_to_delete:
        statements.append(
            (
                """
            MATCH (c:Character:Entity)
            WHERE c.name IN $char_names_to_delete
            DETACH DELETE c
            """,
                {"char_names_to_delete": list(chars_to_delete)},
            )
        )

    for char_name, profile_dict in profiles_data.items():
        if not isinstance(profile_dict, dict):
            logger.warning(f"Skipping invalid profile for '{char_name}' (not a dict).")
            continue

        char_direct_props = {
            k: v
            for k, v in profile_dict.items()
            if isinstance(v, (str, int, float, bool)) and k != "name"
        }

        statements.append(
            (
                """
            MERGE (c:Entity {name: $char_name_val})
            ON CREATE SET
                c:Character,
                c += $props,
                c.created_ts = timestamp()
            ON MATCH SET
                c:Character, 
                c += $props, 
                c.updated_ts = timestamp()
            """,
                {"char_name_val": char_name, "props": char_direct_props},
            )
        )

        statements.append(
            (
                """
            MATCH (ni:NovelInfo:Entity {id: $novel_id_param})
            MATCH (c:Character:Entity {name: $char_name_val})
            MERGE (ni)-[:HAS_CHARACTER]->(c)
            """,
                {"novel_id_param": novel_id_param, "char_name_val": char_name},
            )
        )

        current_profile_traits: Set[str] = {
            str(t).strip()
            for t in profile_dict.get("traits", [])
            if isinstance(t, str) and str(t).strip()
        }

        statements.append(
            (
                """
            MATCH (c:Character:Entity {name: $char_name_val})-[r:HAS_TRAIT]->(t:Trait:Entity)
            WHERE NOT t.name IN $current_traits_list
            DELETE r
            """,
                {
                    "char_name_val": char_name,
                    "current_traits_list": list(current_profile_traits),
                },
            )
        )

        if current_profile_traits:
            statements.append(
                (
                    """
                MATCH (c:Character:Entity {name: $char_name_val})
                UNWIND $current_traits_list AS trait_name_val
                MERGE (t:Entity {name: trait_name_val})
                    ON CREATE SET t:Trait, t.created_ts = timestamp()
                    ON MATCH SET t:Trait
                MERGE (c)-[:HAS_TRAIT]->(t)
                """,
                    {
                        "char_name_val": char_name,
                        "current_traits_list": list(current_profile_traits),
                    },
                )
            )

        statements.append(
            (
                """
            MATCH (c:Character:Entity {name: $char_name_val})-[r:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
            DETACH DELETE dev, r
            """,
                {"char_name_val": char_name},
            )
        )

        for key, value_str in profile_dict.items():
            if (
                key.startswith("development_in_chapter_")
                and isinstance(value_str, str)
                and value_str.strip()
            ):
                try:
                    chap_num_int = int(key.split("_")[-1])
                    dev_event_summary = value_str.strip()
                    dev_event_id = (
                        f"dev_{char_name}_ch{chap_num_int}_{hash(dev_event_summary)}"
                    )

                    dev_event_props = {
                        "id": dev_event_id,
                        "summary": dev_event_summary,
                        KG_NODE_CHAPTER_UPDATED: chap_num_int,
                        KG_IS_PROVISIONAL: profile_dict.get(
                            f"source_quality_chapter_{chap_num_int}"
                        )
                        == "provisional_from_unrevised_draft",
                    }

                    statements.append(
                        (
                            """
                        MATCH (c:Character:Entity {name: $char_name_val})
                        CREATE (dev:Entity:DevelopmentEvent)
                        SET dev = $props, dev.created_ts = timestamp()
                        CREATE (c)-[:DEVELOPED_IN_CHAPTER]->(dev)
                        """,
                            {"char_name_val": char_name, "props": dev_event_props},
                        )
                    )
                except ValueError:
                    logger.warning(
                        f"Could not parse chapter number from dev key: {key} for char {char_name}"
                    )

        profile_defined_rels = profile_dict.get("relationships", {})
        target_chars_in_profile_rels: Set[str] = set()
        if isinstance(profile_defined_rels, dict):
            target_chars_in_profile_rels = {
                str(k).strip() for k in profile_defined_rels.keys() if str(k).strip()
            }

        # Corrected DYNAMIC_REL deletion: Only delete profile-managed relationships
        # whose targets are no longer in the current profile's relationship list.
        statements.append(
            (
                """
            MATCH (c1:Character:Entity {name: $char_name_val})-[r:DYNAMIC_REL]->(c2:Character:Entity)
            WHERE r.source_profile_managed = TRUE AND NOT c2.name IN $target_chars_list
            DELETE r
            """,
                {
                    "char_name_val": char_name,
                    "target_chars_list": list(target_chars_in_profile_rels),
                },
            )
        )

        if isinstance(profile_defined_rels, dict):
            for target_char_name_str, rel_detail in profile_defined_rels.items():
                target_char_name = str(target_char_name_str).strip()
                if not target_char_name:
                    continue

                rel_type_str = "RELATED_TO"
                # Ensure rel_cypher_props is re-initialized for each relationship
                rel_cypher_props = {
                    "source_profile_managed": True,
                    "confidence": 1.0,
                }

                chapter_added_val = config.KG_PREPOPULATION_CHAPTER_NUM
                if isinstance(rel_detail, dict) and "chapter_added" in rel_detail:
                    try:
                        chapter_added_val = int(rel_detail["chapter_added"])
                    except ValueError:
                        pass
                rel_cypher_props["chapter_added"] = chapter_added_val

                if isinstance(rel_detail, str) and rel_detail.strip():
                    rel_cypher_props["description"] = rel_detail.strip()
                    if rel_detail.isupper() and " " not in rel_detail:
                        rel_type_str = rel_detail
                elif isinstance(rel_detail, dict):
                    rel_type_str = (
                        str(rel_detail.get("type", rel_type_str))
                        .upper()
                        .replace(" ", "_")
                    )
                    for k_rel, v_rel in rel_detail.items():
                        if (
                            isinstance(v_rel, (str, int, float, bool))
                            and k_rel != "type"
                            and k_rel != "chapter_added"
                        ):  # chapter_added handled above
                            rel_cypher_props[k_rel] = v_rel

                rel_cypher_props[KG_IS_PROVISIONAL] = (
                    profile_dict.get(f"source_quality_chapter_{chapter_added_val}")
                    == "provisional_from_unrevised_draft"
                )

                statements.append(
                    (
                        """
                    MATCH (s:Character:Entity {name: $subject_param})
                    MATCH (o:Character:Entity {name: $object_param})
                    MERGE (s)-[r:DYNAMIC_REL {type: $predicate_param, chapter_added: $chapter_added_val }]->(o)
                    ON CREATE SET r = $props_param, r.created_ts = timestamp()
                    ON MATCH SET  r = $props_param, r.updated_ts = timestamp()
                    """,
                        {
                            "subject_param": char_name,
                            "object_param": target_char_name,
                            "predicate_param": rel_type_str,
                            "chapter_added_val": chapter_added_val,
                            "props_param": rel_cypher_props,
                        },
                    )
                )

    statements.append(
        (
            """
        MATCH (t:Trait:Entity)
        WHERE NOT EXISTS((:Character:Entity)-[:HAS_TRAIT]->(t))
        DETACH DELETE t
        """,
            {},
        )
    )

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info(
            f"Successfully synchronized {len(all_input_char_names)} character profiles to Neo4j."
        )
        return True
    except Exception as e:
        logger.error(f"Error synchronizing character profiles: {e}", exc_info=True)
        return False


async def get_character_profiles_from_db() -> Dict[str, CharacterProfile]:
    logger.info("Loading decomposed character profiles from Neo4j...")
    profiles_data: Dict[str, CharacterProfile] = {}

    char_query = "MATCH (c:Character:Entity) RETURN c"
    char_results = await neo4j_manager.execute_read_query(char_query)

    if not char_results:
        logger.info("No character profiles found in Neo4j.")
        return {}

    for record in char_results:
        char_node = record.get("c")
        if not char_node:
            continue
        char_name = char_node.get("name")
        if not char_name:
            continue

        profile = dict(char_node)
        profile.pop("name", None)
        profile.pop("created_ts", None)
        profile.pop("updated_ts", None)

        traits_query = "MATCH (:Character:Entity {name: $char_name})-[:HAS_TRAIT]->(t:Trait:Entity) RETURN t.name AS trait_name"
        trait_results = await neo4j_manager.execute_read_query(
            traits_query, {"char_name": char_name}
        )
        profile["traits"] = sorted(
            [tr["trait_name"] for tr in trait_results if tr and tr["trait_name"]]
        )

        rels_query = """
        MATCH (:Character:Entity {name: $char_name})-[r:DYNAMIC_REL]->(target:Character:Entity)
        WHERE r.source_profile_managed = TRUE // Only fetch relationships managed by profiles
        RETURN target.name AS target_name, properties(r) AS rel_props
        """
        rel_results = await neo4j_manager.execute_read_query(
            rels_query, {"char_name": char_name}
        )
        relationships = {}
        if rel_results:
            for rel_rec in rel_results:
                target_name = rel_rec.get("target_name")
                rel_props_full = rel_rec.get("rel_props", {})
                rel_props_cleaned = {
                    k: v
                    for k, v in rel_props_full.items()
                    if k
                    not in [
                        "created_ts",
                        "updated_ts",
                        "source_profile_managed",
                        "chapter_added",
                    ]
                }
                # Restore type if it was part of the key props for merge
                if "type" in rel_props_full:
                    rel_props_cleaned["type"] = rel_props_full["type"]
                # Restore chapter_added from the rel_props_full if it was there
                if "chapter_added" in rel_props_full:
                    rel_props_cleaned["chapter_added"] = rel_props_full["chapter_added"]

                if target_name:
                    relationships[target_name] = rel_props_cleaned
        profile["relationships"] = relationships

        dev_query = (
            "MATCH (:Character:Entity {name: $char_name})-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)\n"
            f"RETURN dev.summary AS summary, dev.{KG_NODE_CHAPTER_UPDATED} AS chapter, dev.{KG_IS_PROVISIONAL} AS is_provisional, dev.id as dev_id\n"
            "ORDER BY dev.chapter_updated ASC"
        )
        dev_results = await neo4j_manager.execute_read_query(
            dev_query, {"char_name": char_name}
        )
        if dev_results:
            for dev_rec in dev_results:
                chapter_num = dev_rec.get("chapter")
                summary = dev_rec.get("summary")
                if chapter_num is not None and summary is not None:
                    dev_key = f"development_in_chapter_{chapter_num}"
                    profile[dev_key] = summary
                    if dev_rec.get(KG_IS_PROVISIONAL):
                        profile[f"source_quality_chapter_{chapter_num}"] = (
                            "provisional_from_unrevised_draft"
                        )

        profiles_data[char_name] = CharacterProfile.from_dict(char_name, profile)

    logger.info(
        f"Successfully loaded and recomposed {len(profiles_data)} character profiles from Neo4j."
    )
    return profiles_data


async def get_character_info_for_snippet_from_db(
    char_name: str, chapter_limit: int
) -> Optional[Dict[str, Any]]:
    query = """
    MATCH (c:Character:Entity {name: $char_name_param})
    
    OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev_np:DevelopmentEvent:Entity)
    WHERE dev_np.chapter_updated <= $chapter_limit_param AND (dev_np.is_provisional IS NULL OR dev_np.is_provisional = FALSE)
    WITH c, dev_np ORDER BY dev_np.chapter_updated DESC
    WITH c, HEAD(COLLECT(dev_np)) AS latest_non_provisional_dev_event

    OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev_p:DevelopmentEvent:Entity)
    WHERE dev_p.chapter_updated <= $chapter_limit_param AND dev_p.is_provisional = TRUE
    WITH c, latest_non_provisional_dev_event, dev_p ORDER BY dev_p.chapter_updated DESC
    WITH c, latest_non_provisional_dev_event, HEAD(COLLECT(dev_p)) AS latest_provisional_dev_event

    WITH c,
         CASE
           WHEN latest_provisional_dev_event IS NOT NULL AND 
                (latest_non_provisional_dev_event IS NULL OR latest_provisional_dev_event.chapter_updated >= latest_non_provisional_dev_event.chapter_updated) 
           THEN latest_provisional_dev_event
           ELSE latest_non_provisional_dev_event
         END AS most_current_dev_event

    OPTIONAL MATCH (c)-[any_rel:DYNAMIC_REL]-(:Entity) 
    WHERE any_rel.is_provisional = TRUE AND any_rel.chapter_added <= $chapter_limit_param 
    
    OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(any_prov_dev_direct:DevelopmentEvent:Entity)
    WHERE any_prov_dev_direct.chapter_updated <= $chapter_limit_param AND any_prov_dev_direct.is_provisional = TRUE
    
    RETURN c.description AS description,
           c.status AS current_status,
           most_current_dev_event.summary AS most_recent_development_note,
           (c.is_provisional = TRUE OR any_rel IS NOT NULL OR any_prov_dev_direct IS NOT NULL) AS is_provisional_overall
    LIMIT 1
    """
    params = {"char_name_param": char_name, "chapter_limit_param": chapter_limit}
    try:
        result = await neo4j_manager.execute_read_query(query, params)
        if result and result[0]:
            record = result[0]
            dev_note = record.get("most_recent_development_note", "N/A")

            return {
                "description": record.get("description"),
                "current_status": record.get("current_status"),
                "most_recent_development_note": dev_note,
                "is_provisional_overall": record.get("is_provisional_overall", False),
            }
        logger.debug(
            f"No detailed snippet info found for character '{char_name}' up to chapter {chapter_limit}."
        )
    except Exception as e:
        logger.error(
            f"Error fetching character info for snippet ({char_name}): {e}",
            exc_info=True,
        )
    return None
