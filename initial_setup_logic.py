# initial_setup_logic.py
import asyncio
import json
import logging
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import config
import utils
from kg_maintainer.models import CharacterProfile, WorldItem
from yaml_parser import load_yaml_file
from story_models import UserStoryInputModel
from llm_interface import llm_service
from prompt_renderer import render_prompt

logger = logging.getLogger(__name__)

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
            if is_list and isinstance(value, list):
                return value, usage_data
            if not is_list and isinstance(value, str):
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
        "logline": config.MARKDOWN_FILL_IN_PLACEHOLDER,
        "inciting_incident": config.MARKDOWN_FILL_IN_PLACEHOLDER,
        "central_conflict": config.MARKDOWN_FILL_IN_PLACEHOLDER,
        "stakes": config.MARKDOWN_FILL_IN_PLACEHOLDER,
        "key_plot_points": [
            f"{config.MARKDOWN_FILL_IN_PLACEHOLDER}"
            for _ in range(num_default_plot_points)
        ],
        "narrative_style": config.MARKDOWN_FILL_IN_PLACEHOLDER,
        "tone": config.MARKDOWN_FILL_IN_PLACEHOLDER,
        "pacing": config.MARKDOWN_FILL_IN_PLACEHOLDER,
        "is_default": True,
        "source": "default_fallback",
    }


def create_default_characters(protagonist_name: str) -> Dict[str, CharacterProfile]:
    """Creates a default character profile for the protagonist."""
    profile = CharacterProfile(name=protagonist_name)
    profile.description = config.MARKDOWN_FILL_IN_PLACEHOLDER
    profile.updates["role"] = "protagonist"
    return {protagonist_name: profile}


def create_default_world() -> Dict[str, Dict[str, WorldItem]]:
    """Creates a default world-building structure."""
    world_data: Dict[str, Dict[str, WorldItem]] = {
        "_overview_": {
            "_overview_": WorldItem.from_dict(
                "_overview_",
                "_overview_",
                {"description": config.CONFIGURED_SETTING_DESCRIPTION},
            )
        },
        "is_default": True,
        "source": "default_fallback",
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
            config.MARKDOWN_FILL_IN_PLACEHOLDER: WorldItem.from_dict(
                cat_key,
                config.MARKDOWN_FILL_IN_PLACEHOLDER,
                {"description": config.MARKDOWN_FILL_IN_PLACEHOLDER},
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

    plot_points = plot_outline.get("key_plot_points", [])
    fill_in_count = sum(1 for p in plot_points if utils._is_fill_in(p))
    needed_plot_points = max(
        0,
        config.TARGET_PLOT_POINTS_INITIAL_GENERATION
        - (len(plot_points) - fill_in_count),
    )

    if needed_plot_points > 0:
        tasks["key_plot_points"] = _bootstrap_field(
            "key_plot_points",
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
        if field == "key_plot_points":
            new_points = value
            final_points = [
                p
                for p in plot_outline.get("key_plot_points", [])
                if not utils._is_fill_in(p)
            ]
            final_points.extend(new_points)
            plot_outline["key_plot_points"] = final_points[
                : config.TARGET_PLOT_POINTS_INITIAL_GENERATION
            ]
        elif value:
            plot_outline[field] = value

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
                character_profiles[name].traits = value
            elif field == "status":
                character_profiles[name].status = value
            else:
                character_profiles[name].updates[field] = value
        character_profiles[name].updates["source"] = "bootstrapped"

    return character_profiles, usage_data if usage_data["total_tokens"] > 0 else None


async def bootstrap_world(
    world_building: Dict[str, Any], plot_outline: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
    """Fills in missing pieces of world building."""
    tasks: Dict[Tuple, Coroutine] = {}
    usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    # Handle overview separately
    if "_overview_" in world_building and "_overview_" in world_building["_overview_"]:
        overview_item = world_building["_overview_"]["_overview_"]
        if isinstance(overview_item, WorldItem) and utils._is_fill_in(
            overview_item.properties.get("description")
        ):
            tasks[("_overview_", "_overview_", "description")] = _bootstrap_field(
                "description",
                {"world_item": overview_item.to_dict(), "plot_outline": plot_outline},
                "bootstrapper/fill_world_item_field.j2",
            )

    for category, items in world_building.items():
        if not isinstance(items, dict) or category == "_overview_":
            continue
        for name, item in items.items():
            if not isinstance(item, WorldItem):
                continue

            if utils._is_fill_in(name):
                # This is a placeholder item that needs a name
                context = {"world_item": item.to_dict(), "plot_outline": plot_outline}
                tasks[(category, name, "name")] = _bootstrap_field(
                    "name", context, "bootstrapper/fill_world_item_field.j2"
                )

            for prop_name, prop_value in item.properties.items():
                if utils._is_fill_in(prop_value):
                    context = {
                        "world_item": item.to_dict(),
                        "plot_outline": plot_outline,
                    }
                    tasks[(category, name, prop_name)] = _bootstrap_field(
                        prop_name, context, "bootstrapper/fill_world_item_field.j2"
                    )

    if not tasks:
        return world_building, None

    results = await asyncio.gather(*tasks.values())
    task_keys = list(tasks.keys())

    new_items_to_add: Dict[str, Dict[str, WorldItem]] = {}
    items_to_remove: Dict[str, List[str]] = {}

    for i, (value, usage) in enumerate(results):
        category, name, prop_name = task_keys[i]
        if usage:
            for k, v in usage.items():
                usage_data[k] = usage_data.get(k, 0) + v
        if value:
            if prop_name == "name":
                # This was a placeholder item, we're giving it a real name
                original_item = world_building[category][name]
                new_item = WorldItem.from_dict(
                    category, value, original_item.properties
                )
                new_items_to_add.setdefault(category, {})[value] = new_item
                items_to_remove.setdefault(category, []).append(name)
            else:
                world_building[category][name].properties[prop_name] = value
                world_building[category][name].properties["source"] = "bootstrapped"

    # Post-processing for renamed items
    for category, names in items_to_remove.items():
        for name in names:
            del world_building[category][name]
    for category, items in new_items_to_add.items():
        world_building[category].update(items)

    world_building["is_default"] = False
    world_building["source"] = "bootstrapped"
    return world_building, usage_data if usage_data["total_tokens"] > 0 else None
