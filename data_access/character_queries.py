# data_access/character_queries.py
import contextlib
from typing import Any

import kg_constants as kg_keys
import structlog
from async_lru import alru_cache  # type: ignore
from config import settings
from core.db_manager import neo4j_manager
from kg_constants import KG_IS_PROVISIONAL, KG_NODE_CHAPTER_UPDATED
from kg_maintainer.models import CharacterProfile
from neo4j.exceptions import ServiceUnavailable  # type: ignore

import utils

from .cypher_builders.character_cypher import (
    TRAIT_NAME_TO_CANONICAL,
    generate_character_node_cypher,
)

# Mapping from normalized character names to canonical display names
CHAR_NAME_TO_CANONICAL: dict[str, str] = {}


def resolve_character_name(name: str) -> str:
    """Return canonical character name for a display variant."""
    if not name:
        return name
    return CHAR_NAME_TO_CANONICAL.get(utils._normalize_for_id(name), name)


logger = structlog.get_logger(__name__)


async def _get_existing_character_names() -> set[str]:
    """Return current character names from the database."""
    try:
        records = await neo4j_manager.execute_read_query(
            "MATCH (c:Character:Entity)"
            " WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE"
            " RETURN c.name AS name"
        )
        return {record["name"] for record in records if record and record.get("name")}
    except Exception as exc:  # pragma: no cover - log and re-raise
        logger.error(
            "Failed to retrieve existing character names from DB: %s",
            exc,
            exc_info=True,
        )
        raise


def _build_dev_event_statements(
    char_name: str, profile_dict: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Return Cypher statements for development events defined in ``profile_dict``."""
    statements: list[tuple[str, dict[str, Any]]] = []
    for key, value_str in profile_dict.items():
        if (
            key.startswith(kg_keys.DEVELOPMENT_PREFIX)
            and isinstance(value_str, str)
            and value_str.strip()
        ):
            try:
                chap_num_int = kg_keys.parse_development_key(key)
                if chap_num_int is None:
                    continue
                dev_event_summary = value_str.strip()
                dev_event_id = f"dev_{utils._normalize_for_id(char_name)}_ch{chap_num_int}_{hash(dev_event_summary)}"
                dev_event_props = {
                    "id": dev_event_id,
                    "summary": dev_event_summary,
                    KG_NODE_CHAPTER_UPDATED: chap_num_int,
                    KG_IS_PROVISIONAL: profile_dict.get(
                        kg_keys.source_quality_key(chap_num_int)
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
            except (ValueError, IndexError):
                logger.warning(
                    "Could not parse chapter number from dev key: %s for char %s",
                    key,
                    char_name,
                )
    return statements


def _build_relationship_statements(
    char_name: str, profile_dict: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Return Cypher statements for profile-defined relationships."""
    statements: list[tuple[str, dict[str, Any]]] = []
    profile_defined_rels = profile_dict.get("relationships", {})
    target_chars_in_profile_rels: set[str] = set()
    if isinstance(profile_defined_rels, dict):
        target_chars_in_profile_rels = {
            str(k).strip() for k in profile_defined_rels if str(k).strip()
        }

    statements.append(
        (
            """
            MATCH (c1:Character:Entity {name: $char_name_val})-[r:DYNAMIC_REL]->(c2:Entity)
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
            rel_cypher_props = {"source_profile_managed": True, "confidence": 1.0}

            chapter_added_val = settings.KG_PREPOPULATION_CHAPTER_NUM
            if isinstance(rel_detail, dict) and "chapter_added" in rel_detail:
                with contextlib.suppress(ValueError, TypeError):
                    chapter_added_val = int(rel_detail["chapter_added"])
            rel_cypher_props["chapter_added"] = chapter_added_val

            if isinstance(rel_detail, str) and rel_detail.strip():
                rel_cypher_props["description"] = rel_detail.strip()
                if rel_detail.isupper() and " " not in rel_detail:
                    rel_type_str = rel_detail
            elif isinstance(rel_detail, dict):
                rel_type_str = (
                    str(rel_detail.get("type", rel_type_str)).upper().replace(" ", "_")
                )
                for k_rel, v_rel in rel_detail.items():
                    if isinstance(v_rel, str | int | float | bool) and k_rel not in {
                        "type",
                        "chapter_added",
                    }:
                        rel_cypher_props[k_rel] = v_rel

            rel_cypher_props[KG_IS_PROVISIONAL] = (
                profile_dict.get(kg_keys.source_quality_key(chapter_added_val))
                == "provisional_from_unrevised_draft"
            )

            statements.append(
                (
                    """
                    MATCH (s:Character:Entity {name: $subject_param})
                    MERGE (o:Entity {name: $object_param})
                        ON CREATE SET o.description = 'Auto-created via relationship from ' + $subject_param, o.created_ts = timestamp()
                    MERGE (s)-[r:DYNAMIC_REL {type: $predicate_param, chapter_added: $chapter_added_val }]->(o)
                    ON CREATE SET r = $props_param, r.created_ts = timestamp()
                    ON MATCH SET  r += $props_param, r.updated_ts = timestamp()
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
    return statements


async def sync_characters(
    profiles: dict[str, CharacterProfile],
    chapter_number: int,
    full_sync: bool = False,
) -> bool:
    """Persist character data to Neo4j."""
    if full_sync:
        profile_dicts = {k: v.to_dict() for k, v in profiles.items()}
        return await sync_full_state_from_object_to_db(profile_dicts)

    statements: list[tuple[str, dict[str, Any]]] = []
    for profile in profiles.values():
        statements.extend(generate_character_node_cypher(profile, chapter_number))

    try:
        if statements:
            await neo4j_manager.execute_cypher_batch(statements)
        logger.info(
            "Persisted %d character updates for chapter %d.",
            len(profiles),
            chapter_number,
        )
        for profile in profiles.values():
            CHAR_NAME_TO_CANONICAL[utils._normalize_for_id(profile.name)] = profile.name
        return True
    except Exception as exc:  # pragma: no cover - log and return failure
        logger.error(
            "Error persisting character updates for chapter %d: %s",
            chapter_number,
            exc,
            exc_info=True,
        )
        return False


async def sync_full_state_from_object_to_db(profiles_data: dict[str, Any]) -> bool:
    """Persist the full set of character profiles to Neo4j."""

    logger.info("Synchronizing character profiles to Neo4j (non-destructive)...")

    novel_id_param = settings.MAIN_NOVEL_INFO_NODE_ID
    statements: list[tuple[str, dict[str, Any]]] = []

    all_input_char_names: set[str] = set(profiles_data.keys())

    try:
        existing_db_char_names = await _get_existing_character_names()
    except Exception:
        return False

    chars_to_delete = existing_db_char_names - all_input_char_names
    if chars_to_delete:
        statements.append(
            (
                """
            MATCH (c:Character:Entity)
            WHERE c.name IN $char_names_to_delete
            SET c.is_deleted = TRUE
            """,
                {"char_names_to_delete": list(chars_to_delete)},
            )
        )

    for char_name, profile_dict in profiles_data.items():
        if not isinstance(profile_dict, dict):
            logger.warning("Skipping invalid profile for '%s' (not a dict).", char_name)
            continue

        char_direct_props = {
            k: v
            for k, v in profile_dict.items()
            if isinstance(v, str | int | float | bool) and k != "name"
        }
        char_direct_props["is_deleted"] = False

        statements.append(
            (
                """
            MERGE (c:Character:Entity {name: $char_name_val})
            ON CREATE SET
                c += $props,
                c.created_ts = timestamp()
            ON MATCH SET
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

        current_profile_traits: set[str] = {
            utils.normalize_trait_name(str(t))
            for t in profile_dict.get("traits", [])
            if isinstance(t, str) and utils.normalize_trait_name(str(t))
        }
        for trait in current_profile_traits:
            TRAIT_NAME_TO_CANONICAL[trait] = trait

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
                MERGE (t:Trait:Entity {name: trait_name_val})
                    ON CREATE SET t.created_ts = timestamp()
                MERGE (c)-[r:HAS_TRAIT]->(t)
                ON CREATE SET r.chapter_added = $prepopulation_chapter_num
                ON MATCH SET r.chapter_added = COALESCE(r.chapter_added, $prepopulation_chapter_num)
                """,
                    {
                        "char_name_val": char_name,
                        "current_traits_list": list(current_profile_traits),
                        "prepopulation_chapter_num": settings.KG_PREPOPULATION_CHAPTER_NUM,
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

        statements.extend(_build_dev_event_statements(char_name, profile_dict))
        statements.extend(_build_relationship_statements(char_name, profile_dict))

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
        for name in all_input_char_names:
            CHAR_NAME_TO_CANONICAL[utils._normalize_for_id(name)] = name
        return True
    except Exception as e:
        logger.error(f"Error synchronizing character profiles: {e}", exc_info=True)
        return False


@alru_cache(maxsize=128)
async def get_character_profile_by_name(name: str) -> CharacterProfile | None:
    """Retrieve a single ``CharacterProfile`` from Neo4j by character name."""
    canonical_name = resolve_character_name(name)
    logger.info("Loading character profile '%s' from Neo4j...", canonical_name)

    query = (
        "MATCH (c:Character:Entity {name: $name})"
        " WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE"
        " RETURN c"
    )
    results = await neo4j_manager.execute_read_query(query, {"name": canonical_name})
    if not results or not results[0].get("c"):
        logger.info("No character profile found for '%s'.", canonical_name)
        return None

    char_node = results[0]["c"]
    profile: dict[str, Any] = dict(char_node)
    profile.pop("name", None)
    profile.pop("created_ts", None)
    profile.pop("updated_ts", None)

    traits_query = (
        "MATCH (:Character:Entity {name: $char_name})-[:HAS_TRAIT]->(t:Trait:Entity)"
        " RETURN t.name AS trait_name"
    )
    trait_results = await neo4j_manager.execute_read_query(
        traits_query, {"char_name": name}
    )
    profile["traits"] = sorted(
        [tr["trait_name"] for tr in trait_results if tr and tr.get("trait_name")]
    )

    rels_query = """
        MATCH (:Character:Entity {name: $char_name})-[r:DYNAMIC_REL]->(target:Entity)
        WHERE r.source_profile_managed = TRUE
        RETURN target.name AS target_name, properties(r) AS rel_props
    """
    rel_results = await neo4j_manager.execute_read_query(
        rels_query, {"char_name": name}
    )
    relationships: dict[str, Any] = {}
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
            if "type" in rel_props_full:
                rel_props_cleaned["type"] = rel_props_full["type"]
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
    dev_results = await neo4j_manager.execute_read_query(dev_query, {"char_name": name})
    if dev_results:
        for dev_rec in dev_results:
            chapter_num = dev_rec.get("chapter")
            summary = dev_rec.get("summary")
            if chapter_num is not None and summary is not None:
                dev_key = kg_keys.development_key(chapter_num)
                profile[dev_key] = summary
                if dev_rec.get(KG_IS_PROVISIONAL):
                    profile[kg_keys.source_quality_key(chapter_num)] = (
                        "provisional_from_unrevised_draft"
                    )

    return CharacterProfile.from_dict(name, profile)


@alru_cache(maxsize=128)
async def get_all_character_names() -> list[str]:
    """Return a list of all character names from Neo4j."""
    query = (
        "MATCH (c:Character:Entity) "
        "WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE "
        "RETURN c.name AS name ORDER BY c.name"
    )
    results = await neo4j_manager.execute_read_query(query)
    return [record["name"] for record in results if record.get("name")]


def _extract_base_profile_data(char_node_data: dict[str, Any] | None) -> dict[str, Any]:
    """Extracts and cleans basic properties from the character node data."""
    if not char_node_data:
        return {}

    profile_dict = dict(char_node_data)
    profile_dict.pop("name", None)  # Name is handled as the key in the main function
    profile_dict.pop("created_ts", None)
    profile_dict.pop("updated_ts", None)
    # Any other common keys to pop can be added here
    return profile_dict


def _process_traits_from_record(record_data: dict[str, Any]) -> list[str]:
    """Processes the 'traits' list from the Cypher query result."""
    traits = record_data.get("traits", []) or []  # Ensure it's a list, default to empty
    # Filter out any None or empty strings that might have slipped through Cypher's CASE WHEN
    processed_traits = [str(t) for t in traits if t and str(t).strip()]
    return sorted(set(processed_traits))  # Ensure uniqueness and sort


def _process_relationships_from_record(record_data: dict[str, Any]) -> dict[str, Any]:
    """Processes the 'rels' list from the Cypher query result into a relationships dictionary."""
    rels_list = record_data.get("rels", []) or []
    relationships: dict[str, Any] = {}
    for rel_rec in rels_list:
        if not isinstance(rel_rec, dict):  # Ensure rel_rec is a dict
            continue
        target = rel_rec.get("target")
        props = rel_rec.get("props", {})
        if not target or not isinstance(
            props, dict
        ):  # Ensure target and props are valid
            continue

        # Clean properties, similar to original logic
        props_cleaned = {
            k: v
            for k, v in props.items()
            if k not in ["created_ts", "updated_ts", "source_profile_managed"]
        }
        # 'type' and 'chapter_added' are often explicitly handled or expected, ensure they are present if in original props
        if "type" in props:
            props_cleaned["type"] = props["type"]
        if "chapter_added" in props:
            props_cleaned["chapter_added"] = props["chapter_added"]

        relationships[str(target)] = props_cleaned
    return relationships


def _process_development_events_from_record(
    record_data: dict[str, Any], profile_dict_to_update: dict[str, Any]
) -> None:
    """Processes 'devs' list from Cypher, updating profile_dict with dev and provisional keys."""
    devs_list = record_data.get("devs", []) or []
    for dev_rec in devs_list:
        if not isinstance(dev_rec, dict):  # Ensure dev_rec is a dict
            continue
        chapter_num = dev_rec.get("chapter")
        summary = dev_rec.get("summary")
        is_provisional = dev_rec.get(
            "prov", False
        )  # Default to False if 'prov' is missing

        if chapter_num is None or summary is None:
            continue

        try:
            # Ensure chapter_num can be converted to int for key generation, though Cypher should return it as such.
            chap_int = int(chapter_num)
            dev_key = kg_keys.development_key(chap_int)
            profile_dict_to_update[dev_key] = str(summary)  # Ensure summary is string
            if is_provisional:
                profile_dict_to_update[kg_keys.source_quality_key(chap_int)] = (
                    "provisional_from_unrevised_draft"
                )
        except (ValueError, TypeError) as e:
            logger.warning(
                f"Could not process development event due to invalid chapter number '{chapter_num}': {e}"
            )


async def get_character_profiles_from_db(
    chapter_limit: int | None = None,
) -> dict[str, CharacterProfile]:
    """Load all character profiles from Neo4j, optionally up to a chapter limit."""
    logger.info(
        "Loading decomposed character profiles from Neo4j%s...",
        f" up to chapter {chapter_limit}" if chapter_limit is not None else "",
    )
    profiles_data: dict[str, CharacterProfile] = {}
    params: dict[str, Any] = {"limit": chapter_limit}
    chapter_filter = ""
    if chapter_limit is not None:
        # Ensure KG_NODE_CHAPTER_UPDATED is correctly referenced for Character nodes if it exists there.
        # The original query implies it might be on `c` or `dev` or `t` (trait chapter_added).
        # For the main character node, if it has a chapter_updated field:
        # chapter_filter = f"AND (c.{KG_NODE_CHAPTER_UPDATED} IS NULL OR c.{KG_NODE_CHAPTER_UPDATED} <= $limit)"
        # However, the query structure seems to filter related entities by chapter,
        # not the character node itself directly by a 'chapter_updated' field.
        # The primary filter on `c` is `c.is_deleted`.
        # The chapter_limit is applied within the collection of traits, rels, and devs.
        pass  # Chapter filter is applied in the sub-collections within Cypher.

    # The Cypher query remains the same as it's complex and optimized for DB interaction.
    # Refactoring focuses on Python-side processing of results.
    query = f"""
    MATCH (c:Character:Entity)
    WHERE (c.is_deleted IS NULL OR c.is_deleted = FALSE) {chapter_filter}

    OPTIONAL MATCH (c)-[t_rel:HAS_TRAIT]->(tr:Trait:Entity)
    WHERE $limit IS NULL OR t_rel.chapter_added <= $limit
    WITH c,
        collect(DISTINCT tr.name) AS traits_collected

    OPTIONAL MATCH (c)-[r_rel:DYNAMIC_REL {{source_profile_managed: TRUE}}]->(target:Entity)
    WHERE $limit IS NULL OR r_rel.chapter_added <= $limit
    WITH c, traits_collected,
        collect(DISTINCT {{target: target.name, props: properties(r_rel)}}) AS rels_collected

    OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
    WHERE ($limit IS NULL OR dev.{KG_NODE_CHAPTER_UPDATED} <= $limit) AND dev.summary IS NOT NULL
    WITH c, traits_collected, rels_collected,
        collect(DISTINCT {{ chapter: dev.{KG_NODE_CHAPTER_UPDATED}, summary: dev.summary, prov: coalesce(dev.{KG_IS_PROVISIONAL}, false) }}) AS devs_collected

    RETURN c,
           traits_collected as traits,
           rels_collected as rels,
           devs_collected as devs
    """
    # Note: The CASE WHEN in Cypher should already filter out NULLs before collect,
    # but adding "WHERE x IS NOT NULL" in the final RETURN list comprehension is a safeguard.

    results = await neo4j_manager.execute_read_query(query, params)

    for record in results:
        char_node_data = record.get("c")
        if not char_node_data:
            continue

        char_name = char_node_data.get("name")
        if not char_name:
            continue

        # Start with base properties from the character node
        profile_dict = _extract_base_profile_data(char_node_data)

        # Process and add traits
        profile_dict["traits"] = _process_traits_from_record(record)

        # Process and add relationships
        profile_dict["relationships"] = _process_relationships_from_record(record)

        # Process development events (updates profile_dict in-place)
        _process_development_events_from_record(record, profile_dict)

        # Conditional addition based on whether any dynamic data was found (if chapter_limit is active)
        # This logic ensures that even if a character node exists, it's only included if it has relevant
        # traits, relationships, or development events within the chapter_limit.
        # If chapter_limit is None, all characters are included.
        has_relevant_dynamic_data = (
            profile_dict.get("traits")
            or profile_dict.get("relationships")
            or any(key.startswith(kg_keys.DEVELOPMENT_PREFIX) for key in profile_dict)
        )

        if chapter_limit is None or has_relevant_dynamic_data:
            CHAR_NAME_TO_CANONICAL[utils._normalize_for_id(char_name)] = char_name
            try:
                profiles_data[char_name] = CharacterProfile.from_dict(
                    char_name, profile_dict
                )
            except (
                Exception
            ) as e:  # Catch potential errors during CharacterProfile instantiation
                logger.error(
                    f"Failed to create CharacterProfile for {char_name} from dict {profile_dict}: {e}",
                    exc_info=True,
                )
                continue  # Skip this profile

    logger.info(
        "Successfully loaded and recomposed %d character profiles from Neo4j%s.",
        len(profiles_data),
        f" up to chapter {chapter_limit}" if chapter_limit is not None else "",
    )
    return profiles_data


async def get_character_info_for_snippet_from_db(
    char_name: str, chapter_limit: int
) -> dict[str, Any] | None:
    """Return character details with history up to ``chapter_limit``.

    Args:
        char_name: Name of the character to fetch.
        chapter_limit: Highest chapter number to consider when collecting data.

    Returns:
        A dictionary of character info or ``None`` if not found.
    """

    canonical_name = resolve_character_name(char_name)
    query = """
    MATCH (c:Character:Entity {name: $char_name_param})
    WHERE c.is_deleted IS NULL OR c.is_deleted = FALSE

    // Subquery to get the most recent non-provisional development event
    CALL (c) {
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
        WHERE dev.chapter_updated <= $chapter_limit_param
          AND (dev.is_provisional IS NULL OR dev.is_provisional = FALSE)
        RETURN dev AS dev_np
        ORDER BY dev.chapter_updated DESC
        LIMIT 1
    }

    // Subquery to get the most recent provisional development event
    CALL (c) {
        OPTIONAL MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
        WHERE dev.chapter_updated <= $chapter_limit_param
          AND dev.is_provisional = TRUE
        RETURN dev AS dev_p
        ORDER BY dev.chapter_updated DESC
        LIMIT 1
    }

    // Subquery to check for the existence of any provisional data related to the character
    CALL (c) {
        RETURN (
            c.is_provisional = TRUE OR
            EXISTS {
                MATCH (c)-[r:DYNAMIC_REL]-(:Entity)
                WHERE r.is_provisional = TRUE AND r.chapter_added <= $chapter_limit_param
            } OR
            EXISTS {
                MATCH (c)-[:DEVELOPED_IN_CHAPTER]->(dev:DevelopmentEvent:Entity)
                WHERE dev.is_provisional = TRUE AND dev.chapter_updated <= $chapter_limit_param
            }
        ) AS is_provisional_flag
    }

    WITH c, dev_np, dev_p, is_provisional_flag

    // Determine the single most current development event
    WITH c, is_provisional_flag,
         CASE
           WHEN dev_p IS NOT NULL AND (dev_np IS NULL OR dev_p.chapter_updated >= dev_np.chapter_updated)
           THEN dev_p
           ELSE dev_np
         END AS most_current_dev_event

    RETURN c.description AS description,
           c.status AS current_status,
           most_current_dev_event,
           is_provisional_flag AS is_provisional_overall
    """
    params = {"char_name_param": canonical_name, "chapter_limit_param": chapter_limit}
    try:
        result = await neo4j_manager.execute_read_query(query, params)
    except ServiceUnavailable as e:
        logger.warning(
            "Neo4j service unavailable when fetching snippet for '%s': %s."
            " Attempting single reconnect.",
            char_name,
            e,
        )
        try:
            await neo4j_manager.connect()
            result = await neo4j_manager.execute_read_query(query, params)
            if result and result[0]:
                record = result[0]
                most_current_dev_event_node = record.get("most_current_dev_event")
                dev_note = (
                    most_current_dev_event_node.get("summary", "N/A")
                    if most_current_dev_event_node
                    else "N/A"
                )

                return {
                    "description": record.get("description"),
                    "current_status": record.get("current_status"),
                    "most_recent_development_note": dev_note,
                    "is_provisional_overall": record.get(
                        "is_provisional_overall", False
                    ),
                }
            logger.debug(
                f"No detailed snippet info found for character '{char_name}' up to chapter {chapter_limit}."
            )
        except Exception as retry_exc:  # pragma: no cover - log and return
            logger.error(
                "Retry after reconnect failed for character '%s': %s",
                char_name,
                retry_exc,
                exc_info=True,
            )
            return None
    except Exception as e:
        logger.error(
            f"Error fetching character info for snippet ({char_name}): {e}",
            exc_info=True,
        )
        return None

    if result and result[0]:
        record = result[0]
        most_current_dev_event_node = record.get("most_current_dev_event")
        dev_note = (
            most_current_dev_event_node.get("summary", "N/A")
            if most_current_dev_event_node
            else "N/A"
        )

        return {
            "description": record.get("description"),
            "current_status": record.get("current_status"),
            "most_recent_development_note": dev_note,
            "is_provisional_overall": record.get("is_provisional_overall", False),
        }
    logger.debug(
        "No detailed snippet info found for character '%s' up to chapter %d.",
        char_name,
        chapter_limit,
    )
    return None


async def find_thin_characters_for_enrichment() -> list[dict[str, Any]]:
    """Finds character nodes that are considered 'thin' (e.g., auto-created stubs)."""
    query = """
    MATCH (c:Character)
    WHERE c.description STARTS WITH 'Auto-created via relationship'
       OR c.description IS NULL
       OR c.description = ''
    RETURN c.name AS name
    LIMIT 20 // Limit to avoid overwhelming the LLM in one cycle
    """
    try:
        results = await neo4j_manager.execute_read_query(query)
        return results if results else []
    except Exception as e:
        logger.error(f"Error finding thin characters: {e}", exc_info=True)
        return []


async def remove_character_trait(char_name: str, trait_name: str) -> bool:
    """Remove a trait from a character node."""
    query = (
        "MATCH (:Character {name: $char_name})-[r:HAS_TRAIT]->(t:Trait {name: $trait_name})"
        " DELETE r"
    )
    cleanup_query = (
        "MATCH (t:Trait {name: $trait_name})"
        " WHERE NOT EXISTS((:Character)-[:HAS_TRAIT]->(t))"
        " DETACH DELETE t"
    )
    try:
        await neo4j_manager.execute_write_query(
            query, {"char_name": char_name, "trait_name": trait_name}
        )
        await neo4j_manager.execute_write_query(
            cleanup_query, {"trait_name": trait_name}
        )
        return True
    except Exception as e:  # pragma: no cover - log but return False
        logger.error(f"Error removing trait '{trait_name}' from '{char_name}': {e}")
        return False
