import asyncio
from collections.abc import Coroutine

import structlog
import utils
from config import settings

from initialization.models import PlotOutline, WorldBuilding, WorldItem

from .common import bootstrap_field

logger = structlog.get_logger(__name__)

WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL: dict[str, str] = {
    "overview": "_overview_",
    "locations": "locations",
    "society": "society",
    "systems": "systems",
    "lore": "lore",
    "history": "history",
    "factions": "factions",
}

WORLD_DETAIL_LIST_INTERNAL_KEYS: list[str] = []


async def generate_world_building_logic(
    world_building: WorldBuilding, plot_outline: PlotOutline
) -> tuple[WorldBuilding, dict[str, int] | None]:
    """Generate complete world-building information."""
    if not world_building or not world_building.data:
        world_building = create_default_world()
    return await bootstrap_world(world_building, plot_outline)


def create_default_world() -> WorldBuilding:
    """Create a default world-building structure."""
    world_data: dict[str, dict[str, WorldItem]] = {
        "_overview_": {
            "_overview_": WorldItem.from_dict(
                "_overview_",
                "_overview_",
                {
                    "description": settings.CONFIGURED_SETTING_DESCRIPTION,
                    "source": "default_overview",
                },
            )
        },
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
            settings.FILL_IN: WorldItem.from_dict(
                cat_key,
                settings.FILL_IN,
                {"description": settings.FILL_IN, "source": "default_placeholder"},
            )
        }

    return WorldBuilding(data=world_data, is_default=True, source="default_fallback")


async def bootstrap_world(
    world_building: WorldBuilding,
    plot_outline: PlotOutline,
) -> tuple[WorldBuilding, dict[str, int] | None]:
    """Fill missing world-building information via LLM."""
    overall_usage_data: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def _accumulate_usage(item_usage: dict[str, int] | None) -> None:
        if item_usage:
            for key, val in item_usage.items():
                overall_usage_data[key] = overall_usage_data.get(key, 0) + val

    # Stage 0: Bootstrap _overview_ description
    if "_overview_" in world_building and "_overview_" in world_building["_overview_"]:
        overview_item_obj = world_building["_overview_"]["_overview_"]
        if isinstance(overview_item_obj, WorldItem) and utils._is_fill_in(
            overview_item_obj.properties.get("description")
        ):
            logger.info("Bootstrapping _overview_ description.")
            desc_value, desc_usage = await bootstrap_field(
                "description",
                {
                    "world_item": overview_item_obj.to_dict(),
                    "plot_outline": plot_outline,
                    "target_category": "_overview_",
                },
                "bootstrapper/fill_world_item_field.j2",
            )
            _accumulate_usage(desc_usage)
            if (
                desc_value
                and isinstance(desc_value, str)
                and not utils._is_fill_in(desc_value)
            ):
                overview_item_obj.properties["description"] = desc_value
                current_source = overview_item_obj.properties.get("source", "")
                if isinstance(current_source, str):
                    overview_item_obj.properties["source"] = (
                        f"{current_source}_descr_bootstrapped"
                        if current_source
                        else "descr_bootstrapped"
                    )
                else:
                    overview_item_obj.properties["source"] = "descr_bootstrapped"

    # Stage 1: Bootstrap names for [Fill-in] items (NEW SEQUENTIAL LOGIC)
    logger.info("Starting sequential name bootstrapping to prevent duplicates...")
    generated_names_this_run: list[str] = []
    items_to_rename: list[dict] = []

    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        # Find all fill-in items for this category
        fill_in_keys = [
            item_name
            for item_name, item_obj in items_dict.items()
            if isinstance(item_obj, WorldItem) and utils._is_fill_in(item_obj.name)
        ]
        
        for fill_in_key in fill_in_keys:
            item_obj = items_dict[fill_in_key]
            logger.info(f"Bootstrapping name for item in category '{category}'...")
            
            context_data = {
                "world_item": item_obj.to_dict(),
                "plot_outline": plot_outline,
                "target_category": category,
                "exclusion_list": generated_names_this_run, # Pass the list of used names
            }

            new_name_value, name_usage = await bootstrap_field(
                "name", context_data, "bootstrapper/fill_world_item_field.j2"
            )
            _accumulate_usage(name_usage)

            if new_name_value and isinstance(new_name_value, str) and not utils._is_fill_in(new_name_value):
                logger.info(f"Generated new name '{new_name_value}' for category '{category}'.")
                generated_names_this_run.append(new_name_value)
                items_to_rename.append({
                    "category": category,
                    "old_name": fill_in_key,
                    "new_name": new_name_value,
                    "item_obj": item_obj
                })
            else:
                logger.warning(f"Failed to bootstrap a valid name for item in category '{category}'.")

    # Now, apply the renames after the loop
    for rename_op in items_to_rename:
        cat = rename_op["category"]
        old_name = rename_op["old_name"]
        new_name = rename_op["new_name"]
        item_obj = rename_op["item_obj"]

        # Remove the old placeholder
        if old_name in world_building[cat]:
            del world_building[cat][old_name]
        
        # Add the new, named item
        new_item = WorldItem.from_dict(cat, new_name, item_obj.properties)
        new_item.properties["source"] = "bootstrapped_name"
        world_building[cat][new_name] = new_item

    # Stage 2: Bootstrap properties for all items (excluding _overview_ top-level)
    property_bootstrap_tasks: dict[tuple[str, str, str], Coroutine] = {}
    for category, items_dict in world_building.items():
        if not isinstance(items_dict, dict) or category == "_overview_":
            continue
        for item_name, item_obj in items_dict.items():
            if not isinstance(item_obj, WorldItem) or utils._is_fill_in(item_name):
                continue
            for prop_name, prop_value in item_obj.properties.items():
                if utils._is_fill_in(prop_value):
                    logger.info(
                        "Identified property '%s' for bootstrapping in item '%s/%s'.",
                        prop_name,
                        category,
                        item_name,
                    )
                    task_key = (category, item_name, prop_name)
                    context_data = {
                        "world_item": item_obj.to_dict(),
                        "plot_outline": plot_outline,
                        "target_category": category,
                    }
                    property_bootstrap_tasks[task_key] = bootstrap_field(
                        prop_name, context_data, "bootstrapper/fill_world_item_field.j2"
                    )

    if property_bootstrap_tasks:
        logger.info(
            "Found %d properties requiring bootstrapping.",
            len(property_bootstrap_tasks),
        )
        property_results_list = await asyncio.gather(*property_bootstrap_tasks.values())
        property_task_keys = list(property_bootstrap_tasks.keys())

        for i, (prop_fill_value, prop_usage) in enumerate(property_results_list):
            _accumulate_usage(prop_usage)
            category, item_name, prop_name_filled = property_task_keys[i]

            target_item = world_building.get(category, {}).get(item_name)
            if not target_item:
                logger.warning(
                    "Item %s/%s not found while trying to update property %s. Skipping.",
                    category,
                    item_name,
                    prop_name_filled,
                )
                continue

            if prop_fill_value is not None and not (
                isinstance(prop_fill_value, str) and utils._is_fill_in(prop_fill_value)
            ):
                logger.info(
                    "Successfully bootstrapped property '%s' for item '%s/%s'.",
                    prop_name_filled,
                    category,
                    item_name,
                )
                target_item.properties[prop_name_filled] = prop_fill_value

                current_source = target_item.properties.get("source", "")
                if isinstance(current_source, str):
                    append_source = f"_prop_{prop_name_filled}_bootstrapped"
                    if append_source not in current_source:
                        target_item.properties["source"] = (
                            f"{current_source}{append_source}"
                            if current_source
                            else append_source.lstrip("_")
                        )
                else:
                    target_item.properties["source"] = (
                        f"prop_{prop_name_filled}_bootstrapped"
                    )
            else:
                logger.warning(
                    "Property bootstrapping for '%s' in '%s/%s' resulted in empty or FILL_IN value.",
                    prop_name_filled,
                    category,
                    item_name,
                )

    if overall_usage_data["total_tokens"] > 0:
        world_building["is_default"] = False  # type: ignore
        current_top_source = world_building.get("source", "")
        if isinstance(current_top_source, str):
            world_building["source"] = (
                f"{current_top_source}_bootstrapped_items"
                if current_top_source and "default" not in current_top_source
                else "bootstrapped_items"
            )  # type: ignore
        else:
            world_building["source"] = "bootstrapped_items"  # type: ignore
        logger.info(
            "World building bootstrapping complete. Marking as not default and source as bootstrapped."
        )

    return world_building, overall_usage_data if overall_usage_data[
        "total_tokens"
    ] > 0 else None
