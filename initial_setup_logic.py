# initial_setup_logic.py
import asyncio
import json
import structlog
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import config
import utils
from kg_maintainer.models import CharacterProfile, WorldItem
from kg_maintainer_agent import KGMaintainerAgent
from yaml_parser import load_yaml_file
from story_models import UserStoryInputModel, user_story_to_objects
from llm_interface import llm_service
from prompt_renderer import render_prompt
from pydantic import ValidationError

logger = structlog.get_logger(__name__)

WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL: Dict[str, str] = {
    "overview": "_overview_",
    "locations": "locations",
    "society": "society",
    "systems": "systems",
    "lore": "lore",
    "history": "history",
    "factions": "factions",
}

WORLD_DETAIL_LIST_INTERNAL_KEYS: List[str] = []


def _load_user_supplied_data() -> Optional[UserStoryInputModel]:
    """Load story elements from YAML if available."""
    data = load_yaml_file(config.USER_STORY_ELEMENTS_FILE_PATH)
    if not data:
        return None
    try:
        return UserStoryInputModel(**data)
    except Exception as exc:  # pragma: no cover - simple validation wrapper
        logger.error("Failed to parse user story YAML: %s", exc)
        return None


async def generate_world_building_logic(
    world_building: Dict[str, Any], plot_outline: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    """Stub world-building generation function."""
    logger.warning("generate_world_building_logic stub called")
    if not world_building:
        world_building = create_default_world()
    return world_building, None


async def _bootstrap_field(
    field_name: str,
    context_data: Dict[str, Any],
    prompt_template_path: str,
    is_list: bool = False,
    list_count: int = 1,
) -> Tuple[Any, Optional[Dict[str, int]]]:
    """
    Surgically calls an LLM to fill a single field or a list of fields,
    and returns the value and the token usage data.
    """
    logger.info(f"Bootstrapping field: '{field_name}'...")
    prompt = render_prompt(
        prompt_template_path,
        {"context": context_data, "field_name": field_name, "list_count": list_count},
    )

    response_text, usage_data = await llm_service.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt,
        temperature=config.Temperatures.INITIAL_SETUP,
        stream_to_disk=False,
        auto_clean_response=True,
    )

    if not response_text.strip():
        logger.warning(
            f"LLM returned empty response for bootstrapping field '{field_name}'."
        )
        return ([] if is_list else ""), usage_data

    try:
        parsed_json = json.loads(response_text)
        if isinstance(parsed_json, dict):
            value = parsed_json.get(field_name)

            if is_list:
                if isinstance(value, list):
                    return value, usage_data
                elif isinstance(value, str):
                    # FIX: Handle the case where LLM returns a comma or newline-separated string for a list.
                    logger.info(
                        f"LLM returned a string for list field '{field_name}'. Parsing string into list."
                    )
                    items = [
                        item.strip().lstrip("-* ").strip()
                        for item in value.replace("\n", ",").split(",")
                        if item.strip()
                    ]
                    return items, usage_data
            elif isinstance(value, str):  # if not is_list
                return value.strip(), usage_data

            logger.warning(
                f"LLM JSON for '{field_name}' had unexpected type. Got: {type(value)}"
            )
        else:
            logger.warning(
                f"LLM response for '{field_name}' was not a JSON object. Response: {response_text[:100]}"
            )
    except json.JSONDecodeError:
        if is_list:
            return (
                [line.strip() for line in response_text.splitlines() if line.strip()],
                usage_data,
            )
        return response_text.strip(), usage_data

    return ([] if is_list else ""), usage_data


def create_default_plot(default_protagonist_name: str) -> Dict[str, Any]:
    """Creates a default plot outline with placeholders."""
    num_default_plot_points = config.TARGET_PLOT_POINTS_INITIAL_GENERATION
    return {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
        "protagonist_name": default_protagonist_name,
        "genre": config.CONFIGURED_GENRE,
        "setting": config.CONFIGURED_SETTING_DESCRIPTION,
        "theme": config.CONFIGURED_THEME,
        "logline": config.FILL_IN,
        "inciting_incident": config.FILL_IN,
        "central_conflict": config.FILL_IN,
        "stakes": config.FILL_IN,
        "plot_points": [f"{config.FILL_IN}" for _ in range(num_default_plot_points)],
        "narrative_style": config.FILL_IN,
        "tone": config.FILL_IN,
        "pacing": config.FILL_IN,
        "is_default": True,
        "source": "default_fallback",
    }


def create_default_characters(protagonist_name: str) -> Dict[str, CharacterProfile]:
    """Creates a default character profile for the protagonist."""
    profile = CharacterProfile(name=protagonist_name)
    profile.description = config.FILL_IN
    profile.updates["role"] = "protagonist"
    return {protagonist_name: profile}


def create_default_world() -> Dict[str, Dict[str, WorldItem]]:
    """Creates a default world-building structure."""
    world_data: Dict[str, Dict[str, WorldItem]] = { # type: ignore
        "_overview_": {
            "_overview_": WorldItem.from_dict(
                "_overview_",
                "_overview_",
                {"description": config.CONFIGURED_SETTING_DESCRIPTION, "source": "default_overview"},
            )
        },
        "is_default": True, # type: ignore
        "source": "default_fallback", # type: ignore
    }

    standard_categories = [
        "locations",
        "society",
        "systems",
        "lore",
        "history",
        "factions",
    ]

    for cat_key in standard_categories:
        world_data[cat_key] = {
            config.FILL_IN: WorldItem.from_dict(
                cat_key,
                config.FILL_IN,
                {"description": config.FILL_IN, "source": "default_placeholder"},
            )
        }

    return world_data


async def bootstrap_plot_outline(
    plot_outline: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    """Fills in missing pieces of the plot outline using surgical LLM calls."""
    tasks: Dict[str, Coroutine] = {}
    usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    fields_to_bootstrap = {
        "title": (
            not plot_outline.get("title")
            or utils._is_fill_in(plot_outline.get("title"))
        ),
        "protagonist_name": (
            not plot_outline.get("protagonist_name")
            or utils._is_fill_in(plot_outline.get("protagonist_name"))
        ),
        "genre": (
            not plot_outline.get("genre")
            or utils._is_fill_in(plot_outline.get("genre"))
        ),
        "setting": (
            not plot_outline.get("setting")
            or utils._is_fill_in(plot_outline.get("setting"))
        ),
        "theme": (
            not plot_outline.get("theme")
            or utils._is_fill_in(plot_outline.get("theme"))
        ),
        "logline": (
            not plot_outline.get("logline")
            or utils._is_fill_in(plot_outline.get("logline"))
        ),
        "inciting_incident": (
            not plot_outline.get("inciting_incident")
            or utils._is_fill_in(plot_outline.get("inciting_incident"))
        ),
        "central_conflict": (
            not plot_outline.get("central_conflict")
            or utils._is_fill_in(plot_outline.get("central_conflict"))
        ),
        "stakes": (
            not plot_outline.get("stakes")
            or utils._is_fill_in(plot_outline.get("stakes"))
        ),
        "narrative_style": (
            not plot_outline.get("narrative_style")
            or utils._is_fill_in(plot_outline.get("narrative_style"))
        ),
        "tone": (
            not plot_outline.get("tone") or utils._is_fill_in(plot_outline.get("tone"))
        ),
        "pacing": (
            not plot_outline.get("pacing")
            or utils._is_fill_in(plot_outline.get("pacing"))
        ),
    }

    for field, needed in fields_to_bootstrap.items():
        if needed:
            tasks[field] = _bootstrap_field(
                field, plot_outline, "bootstrapper/fill_plot_field.j2"
            )

    plot_points = plot_outline.get("plot_points", [])
    fill_in_count = sum(1 for p in plot_points if utils._is_fill_in(p))
    needed_plot_points = max(
        0,
        config.TARGET_PLOT_POINTS_INITIAL_GENERATION
        - (len(plot_points) - fill_in_count),
    )

    if needed_plot_points > 0:
        tasks["plot_points"] = _bootstrap_field(
            "plot_points",
            plot_outline,
            "bootstrapper/fill_plot_points.j2",
            is_list=True,
            list_count=needed_plot_points,
        )

    if not tasks:
        return plot_outline, None

    results = await asyncio.gather(*tasks.values())
    task_keys = list(tasks.keys())

    for i, (value, usage) in enumerate(results):
        field = task_keys[i]
        if usage:
            for k, v in usage.items():
                usage_data[k] = usage_data.get(k, 0) + v
        if field == "plot_points":
            new_points = value
            final_points = [
                p
                for p in plot_outline.get("plot_points", [])
                if not utils._is_fill_in(p)
            ]
            final_points.extend(new_points)
            plot_outline["plot_points"] = final_points[
                : config.TARGET_PLOT_POINTS_INITIAL_GENERATION
            ]
        elif value:
            plot_outline[field] = value

    if usage_data["total_tokens"] > 0:
        plot_outline["is_default"] = False
        plot_outline["source"] = "bootstrapped"
    return plot_outline, usage_data if usage_data["total_tokens"] > 0 else None


async def bootstrap_characters(
    character_profiles: Dict[str, CharacterProfile], plot_outline: Dict[str, Any]
) -> Tuple[Dict[str, CharacterProfile], Optional[Dict[str, int]]]:
    """Fills in missing pieces of character profiles."""
    tasks: Dict[Tuple[str, str], Coroutine] = {}
    usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for name, profile in character_profiles.items():
        context = {"profile": profile.to_dict(), "plot_outline": plot_outline}

        if not profile.description or utils._is_fill_in(profile.description):
            tasks[(name, "description")] = _bootstrap_field(
                "description",
                context,
                "bootstrapper/fill_character_field.j2",
            )

        if not profile.status or utils._is_fill_in(profile.status):
            tasks[(name, "status")] = _bootstrap_field(
                "status",
                context,
                "bootstrapper/fill_character_field.j2",
            )

        trait_fill_count = sum(1 for t in profile.traits if utils._is_fill_in(t))
        if trait_fill_count or not profile.traits:
            tasks[(name, "traits")] = _bootstrap_field(
                "traits",
                context,
                "bootstrapper/fill_character_field.j2",
                is_list=True,
                list_count=max(trait_fill_count, 3),
            )

        if "motivation" in profile.updates and utils._is_fill_in(
            profile.updates["motivation"]
        ):
            tasks[(name, "motivation")] = _bootstrap_field(
                "motivation",
                context,
                "bootstrapper/fill_character_field.j2",
            )

    if not tasks:
        return character_profiles, None

    results = await asyncio.gather(*tasks.values())
    task_keys = list(tasks.keys())

    for i, (value, usage) in enumerate(results):
        name, field = task_keys[i]
        if usage:
            for k, v in usage.items():
                usage_data[k] = usage_data.get(k, 0) + v
        if value:
            if field == "description":
                character_profiles[name].description = value
            elif field == "traits":
                character_profiles[name].traits = value # type: ignore
            elif field == "status":
                character_profiles[name].status = value
            else: # motivation
                character_profiles[name].updates[field] = value
            character_profiles[name].updates["source"] = "bootstrapped"


    if usage_data["total_tokens"] > 0:
        # Mark all affected profiles, not just the last one.
        # This is slightly different from plot, as char profiles are dicts.
        # The "source" update inside the loop handles individual profiles.
        pass

    return character_profiles, usage_data if usage_data["total_tokens"] > 0 else None


async def bootstrap_world(
    world_building: Dict[str, Any], plot_outline: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    """Fills in missing pieces of world building in a multi-stage process."""
    overall_usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _accumulate_usage(item_usage: Optional[Dict[str, int]]) -> None:
        if item_usage:
            for key, val in item_usage.items():
                overall_usage_data[key] = overall_usage_data.get(key, 0) + val

    # Stage 0: Bootstrap _overview_ Description
    if "_overview_" in world_building and "_overview_" in world_building["_overview_"]:
        overview_item_obj = world_building["_overview_"]["_overview_"]
        if isinstance(overview_item_obj, WorldItem) and utils._is_fill_in(
            overview_item_obj.properties.get("description")
        ):
            logger.info("Bootstrapping _overview_ description.")
            desc_value, desc_usage = await _bootstrap_field(
                "description",
                {"world_item": overview_item_obj.to_dict(), "plot_outline": plot_outline, "target_category": "_overview_"},
                "bootstrapper/fill_world_item_field.j2",
            )
            _accumulate_usage(desc_usage)
            if desc_value and isinstance(desc_value, str) and not utils._is_fill_in(desc_value):
                overview_item_obj.properties["description"] = desc_value
                current_source = overview_item_obj.properties.get("source", "")
                if isinstance(current_source, str):
                    overview_item_obj.properties["source"] = f"{current_source}_descr_bootstrapped" if current_source else "descr_bootstrapped"
                else: # If source is not a string (e.g. list, or None), overwrite or handle as per desired logic
                    overview_item_obj.properties["source"] = "descr_bootstrapped"

    # Stage 1: Bootstrap Names for [Fill-in] Items
    name_bootstrap_tasks: Dict[Tuple[str, str], Coroutine] = {}
    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        for item_name, item_obj in items_dict.items():
            if isinstance(item_obj, WorldItem) and utils._is_fill_in(item_obj.name): # item_obj.name is the key `item_name`
                logger.info(f"Identified item for name bootstrapping in category '{category}': Current name '{item_name}'")
                task_key = (category, item_name) # item_name is config.FILL_IN here
                context_data = {"world_item": item_obj.to_dict(), "plot_outline": plot_outline, "target_category": category}
                name_bootstrap_tasks[task_key] = _bootstrap_field(
                    "name", context_data, "bootstrapper/fill_world_item_field.j2"
                )

    if name_bootstrap_tasks:
        logger.info(f"Found {len(name_bootstrap_tasks)} items requiring name bootstrapping.")
        name_results_list = await asyncio.gather(*name_bootstrap_tasks.values())
        name_task_keys = list(name_bootstrap_tasks.keys())

        new_items_to_add_stage1: Dict[str, Dict[str, WorldItem]] = {}
        items_to_remove_stage1: Dict[str, List[str]] = {}

        for i, (new_name_value, name_usage) in enumerate(name_results_list):
            _accumulate_usage(name_usage)
            original_category, original_fill_in_name = name_task_keys[i]

            if new_name_value and isinstance(new_name_value, str) and not utils._is_fill_in(new_name_value) and new_name_value != config.FILL_IN:
                original_item_obj = world_building[original_category][original_fill_in_name]

                # Check for name collision in the same category before adding
                if new_name_value in world_building[original_category] or \
                   new_name_value in new_items_to_add_stage1.get(original_category, {}):
                    logger.warning(
                        f"Name bootstrap for '{original_category}/{original_fill_in_name}' resulted in a duplicate name '{new_name_value}'. Skipping this item."
                    )
                    continue

                logger.info(f"Successfully bootstrapped name for '{original_category}/{original_fill_in_name}': New name is '{new_name_value}'")
                new_item_renamed = WorldItem.from_dict(original_category, new_name_value, original_item_obj.properties)
                new_item_renamed.properties["source"] = "bootstrapped_name" # Overwrite or append as needed

                new_items_to_add_stage1.setdefault(original_category, {})[new_name_value] = new_item_renamed
                items_to_remove_stage1.setdefault(original_category, []).append(original_fill_in_name)
            else:
                logger.warning(
                    f"Name bootstrapping failed for item in category '{original_category}' (original key: '{original_fill_in_name}'). Received: '{new_name_value}'. Item will retain placeholder name."
                )

        # Apply changes from Stage 1 to world_building
        for cat, names_to_remove in items_to_remove_stage1.items():
            for name_key in names_to_remove:
                if name_key in world_building[cat]:
                    del world_building[cat][name_key]
        for cat, new_items_map in new_items_to_add_stage1.items():
            if cat not in world_building:
                world_building[cat] = {}
            world_building[cat].update(new_items_map)
        logger.info("Finished applying name changes from Stage 1 to world_building structure.")

    # Stage 2: Bootstrap Properties for All Items (excluding _overview_ top-level)
    property_bootstrap_tasks: Dict[Tuple[str, str, str], Coroutine] = {}
    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        for item_name, item_obj in items_dict.items():
            if not isinstance(item_obj, WorldItem) or utils._is_fill_in(item_name): # Skip items that still have placeholder name
                continue
            for prop_name, prop_value in item_obj.properties.items():
                if utils._is_fill_in(prop_value):
                    logger.info(f"Identified property '{prop_name}' for bootstrapping in item '{category}/{item_name}'.")
                    task_key = (category, item_name, prop_name)
                    context_data = {"world_item": item_obj.to_dict(), "plot_outline": plot_outline, "target_category": category}
                    property_bootstrap_tasks[task_key] = _bootstrap_field(
                        prop_name, context_data, "bootstrapper/fill_world_item_field.j2"
                    )

    if property_bootstrap_tasks:
        logger.info(f"Found {len(property_bootstrap_tasks)} properties requiring bootstrapping.")
        property_results_list = await asyncio.gather(*property_bootstrap_tasks.values())
        property_task_keys = list(property_bootstrap_tasks.keys())

        for i, (prop_fill_value, prop_usage) in enumerate(property_results_list):
            _accumulate_usage(prop_usage)
            category, item_name, prop_name_filled = property_task_keys[i]

            target_item = world_building.get(category, {}).get(item_name)
            if not target_item:
                logger.warning(f"Item {category}/{item_name} not found while trying to update property {prop_name_filled}. Skipping.")
                continue

            if prop_fill_value is not None and not (isinstance(prop_fill_value, str) and utils._is_fill_in(prop_fill_value)):
                logger.info(f"Successfully bootstrapped property '{prop_name_filled}' for item '{category}/{item_name}'.")
                target_item.properties[prop_name_filled] = prop_fill_value

                current_source = target_item.properties.get("source", "")
                if isinstance(current_source, str):
                    append_source = f"_prop_{prop_name_filled}_bootstrapped"
                    if append_source not in current_source: # Avoid duplicate appends if run multiple times
                         target_item.properties["source"] = f"{current_source}{append_source}" if current_source else append_source.lstrip('_')
                else:
                    target_item.properties["source"] = f"prop_{prop_name_filled}_bootstrapped"
            else:
                logger.warning(f"Property bootstrapping for '{prop_name_filled}' in '{category}/{item_name}' resulted in empty or FILL_IN value.")

    if overall_usage_data["total_tokens"] > 0:
        world_building["is_default"] = False # type: ignore
        current_top_source = world_building.get("source", "")
        if isinstance(current_top_source, str):
             world_building["source"] = f"{current_top_source}_bootstrapped_items" if current_top_source and "default" not in current_top_source else "bootstrapped_items" # type: ignore
        else:
            world_building["source"] = "bootstrapped_items" # type: ignore
        logger.info("World building bootstrapping complete. Marking as not default and source as bootstrapped.")

    return world_building, overall_usage_data if overall_usage_data["total_tokens"] > 0 else None


async def run_genesis_phase() -> Tuple[
    Dict[str, Any],
    Dict[str, CharacterProfile],
    Dict[str, Dict[str, WorldItem]],
    Dict[str, int],
]:
    """Execute Phase 1 of SAGA's pipeline."""

    def _add_usage(total: Dict[str, int], usage: Optional[Dict[str, int]]) -> None:
        if not usage:
            return
        for key, val in usage.items():
            total[key] = total.get(key, 0) + val

    user_data = load_yaml_file(config.USER_STORY_ELEMENTS_FILE_PATH)
    plot_outline: Dict[str, Any]
    character_profiles: Dict[str, CharacterProfile]
    world_building: Dict[str, Dict[str, WorldItem]] # Adjusted type hint for world_building items

    if user_data:
        try:
            model = UserStoryInputModel.model_validate(user_data)
            (
                plot_outline,
                character_profiles,
                world_building_untyped, # Keep original name from model parsing
            ) = user_story_to_objects(model)

            # Ensure world_building is correctly typed after parsing
            world_building = {}
            for k, v in world_building_untyped.items():
                if k in ["is_default", "source"]: # Handle top-level metadata
                    world_building[k] = v # type: ignore
                elif isinstance(v, dict):
                    world_building[k] = {
                        item_k: (item_v if isinstance(item_v, WorldItem) else WorldItem.from_dict(k, item_k, item_v if isinstance(item_v, dict) else {}))
                        for item_k, item_v in v.items()
                    }
                else: # Should not happen with current model, but good for robustness
                    world_building[k] = v # type: ignore


            plot_outline["source"] = "user_supplied_yaml"
            if "source" not in world_building : # type: ignore
                 world_building["source"] = "user_supplied_yaml" # type: ignore
            logger.info("Loaded user story elements from YAML file.")
        except ValidationError as exc:  # pragma: no cover - fallback path
            logger.error("User YAML validation failed: %s", exc)
            user_data = None # type: ignore

    if not user_data: # type: ignore
        logger.info("No valid user YAML found. Using default placeholders.")
        plot_outline = create_default_plot(config.DEFAULT_PROTAGONIST_NAME)
        character_profiles = create_default_characters(plot_outline["protagonist_name"])
        world_building = create_default_world()

    plot_outline, plot_usage = await bootstrap_plot_outline(plot_outline)
    character_profiles, char_usage = await bootstrap_characters(
        character_profiles, plot_outline
    )
    # Ensure world_building is passed correctly, it's Dict[str, Any] due to is_default/source flags
    world_building_typed_for_bootstrap: Dict[str, Any] = world_building
    world_building_typed_for_bootstrap, world_usage = await bootstrap_world(world_building_typed_for_bootstrap, plot_outline)

    # After bootstrapping, ensure the main world_building variable reflects changes
    world_building = world_building_typed_for_bootstrap


    usage_totals: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    _add_usage(usage_totals, plot_usage)
    _add_usage(usage_totals, char_usage)
    _add_usage(usage_totals, world_usage)

    kg_agent = KGMaintainerAgent()
    # Ensure world_building is correctly typed for persist_world
    # It expects Dict[str, Dict[str, WorldItem]] but our world_building also has is_default/source
    world_items_for_kg: Dict[str, Dict[str, WorldItem]] = {
        k: v for k, v in world_building.items() if k not in ["is_default", "source"] and isinstance(v, dict)
    }

    await kg_agent.persist_profiles(
        character_profiles, config.KG_PREPOPULATION_CHAPTER_NUM
    )
    await kg_agent.persist_world(world_items_for_kg, config.KG_PREPOPULATION_CHAPTER_NUM) # type: ignore
    logger.info("Knowledge graph pre-population complete.")

    # Final return needs Dict[str, Dict[str, WorldItem]] for world_building
    # So, we return the filtered version, or adjust downstream expectations.
    # For now, returning the version with metadata, caller might need adjustment or this function's return type.
    # Let's assume the caller can handle the Dict[str, Any] which contains WorldItem dicts + metadata.
    return plot_outline, character_profiles, world_building, usage_totals
