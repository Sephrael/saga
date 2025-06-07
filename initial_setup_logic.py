# initial_setup_logic.py
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

import config
import utils  # For _is_fill_in
from kg_maintainer.models import CharacterProfile, WorldItem
from llm_interface import llm_service
from story_models import UserStoryInputModel
from yaml_parser import load_yaml_file

logger = logging.getLogger(__name__)

PlotOutlineData = Dict[str, Any]
WorldBuildingData = Dict[str, Any]

PLOT_OUTLINE_KEY_MAP = {
    "title": "title",
    "protagonist_name": "protagonist_name",
    "protagonist_description": "protagonist_description",
    "plot_points": "plot_points",
    "character_arc": "character_arc",
    "conflict_summary": "conflict_summary",
    "logline": "logline",
    "setting_description": "setting_description",
    "inciting_incident": "inciting_incident",
    "climax_event_preview": "climax_event_preview",
    "antagonist_name": "antagonist_name",
    "antagonist_description": "antagonist_description",
    "antagonist_motivations": "antagonist_motivations",
    "genre": "genre",
    "theme": "theme",  # Added genre and theme to be fillable
    "stakes": "stakes",
}
PLOT_OUTLINE_LIST_INTERNAL_KEYS = ["plot_points"]

WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL = {
    "overview": "_overview_",
    "locations": "locations",
    "society": "society",
    "systems": "systems",
    "lore": "lore",
    "history": "history",
    "factions": "factions",
}
# This map is crucial for mapping keys from Markdown (normalized) to your agent's internal keys
WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL = {
    "description": "description",
    "atmosphere": "atmosphere",
    "modification_proposal": "modification_proposal",
    "goals": "goals",
    "rules": "rules",
    "key_elements": "key_elements",
    "traits": "traits",
    "primary_setting_description": "description",  # For top-level setting section
    "known_effects": "known_effects",  # For lore items
    "mood": "mood",  # From LLM output
    "time_period": "time_period",  # From LLM output
    "primary_conflict": "primary_conflict",  # From LLM output
    "features": "features",  # From LLM output
    "function": "function",  # For systems from LLM output
    "key_beliefs": "key_beliefs",  # For lore/factions from LLM output
    "key_events": "key_events",  # For lore/history from LLM output
    "key_figures": "key_figures",  # For history from LLM output
    "impact_on_daily_life": "impact_on_daily_life",  # For society from LLM output
    "structure": "structure",  # For society/factions from LLM output
}
WORLD_DETAIL_LIST_INTERNAL_KEYS = [
    "goals",
    "rules",
    "key_elements",
    "traits",
    "key_beliefs",
    "key_events",
    "key_figures",
    "features",
]  # Added features


def _get_val_or_fill_in(
    data_dict: Optional[Dict[str, Any]],
    key: str,
    default_is_fill_in: bool = True,
) -> Any:
    if data_dict is None:
        return config.MARKDOWN_FILL_IN_PLACEHOLDER if default_is_fill_in else ""
    val = data_dict.get(key)
    if val is None or (
        isinstance(val, str) and not val.strip()
    ):  # Empty string is also a "fill-in" case
        return config.MARKDOWN_FILL_IN_PLACEHOLDER if default_is_fill_in else ""
    return val


def _create_default_plot(
    default_protagonist_name: str,
    base_elements: Dict[str, Any],
    unhinged: bool,
) -> PlotOutlineData:
    num_default_plot_points = config.TARGET_PLOT_POINTS_INITIAL_GENERATION
    default_plot: PlotOutlineData = {
        "title": config.DEFAULT_PLOT_OUTLINE_TITLE,
        "protagonist_name": default_protagonist_name,
        "protagonist_description": (
            f"Default protagonist: {default_protagonist_name}, a character facing challenges."
        ),
        "plot_points": [
            f"{config.MARKDOWN_FILL_IN_PLACEHOLDER} - Default Plot Point {i + 1}"
            for i in range(num_default_plot_points)
        ],
        "character_arc": (
            f"Default arc: {default_protagonist_name} learns something important."
        ),
        "setting_description": base_elements.get(
            "setting_description",
            base_elements.get("setting", "A generic place."),
        ),
        "conflict_summary": (
            "Default conflict: the protagonist faces obstacles tied to the theme."
        ),
        "is_default": True,
        "source": "default_fallback",
    }
    default_plot.update(
        {k: v for k, v in base_elements.items() if k in ["genre", "theme"]}
    )
    if unhinged:
        default_plot.update(
            {
                k: base_elements[k]
                for k in [
                    "setting_archetype_used",
                    "protagonist_archetype_used",
                    "conflict_archetype_used",
                ]
                if k in base_elements
            }
        )
    for key_in_map in PLOT_OUTLINE_KEY_MAP.values():
        if key_in_map not in default_plot:
            default_plot[key_in_map] = (
                []
                if key_in_map in PLOT_OUTLINE_LIST_INTERNAL_KEYS
                else config.MARKDOWN_FILL_IN_PLACEHOLDER
            )
    return default_plot


def _load_user_supplied_data() -> Optional[UserStoryInputModel]:
    """Load and validate user-supplied story data."""
    # Assuming USER_STORY_ELEMENTS_FILE_PATH in config will be updated to point to a .yaml file
    # or a new config variable USER_STORY_ELEMENTS_YAML_FILE_PATH will be used.
    # For this change, directly adjusting the expected file extension.
    yaml_file_path = config.USER_STORY_ELEMENTS_FILE_PATH.replace(
        ".md", ".yaml"
    ).replace(".md.example", ".yaml.example")
    if not yaml_file_path.endswith(
        (".yaml", ".yml")
    ):  # Ensure it's a yaml path after replace
        yaml_file_path = (
            os.path.splitext(config.USER_STORY_ELEMENTS_FILE_PATH)[0] + ".yaml"
        )
        logger.info(
            f"Adjusted file path to: {yaml_file_path} from {config.USER_STORY_ELEMENTS_FILE_PATH}"
        )

    user_data = load_yaml_file(yaml_file_path)
    if user_data is None:
        logger.info(
            f"User story elements file '{yaml_file_path}' not found or invalid. "
            "Proceeding with LLM generation or defaults."
        )
        return None
    # load_yaml_file returns {} for empty file, or None for critical parse error / non-dict root
    if not isinstance(user_data, dict) or not user_data:  # Ensure it's a non-empty dict
        logger.warning(
            f"User story elements file '{yaml_file_path}' was empty or invalid. "
            "Using LLM generation or defaults."
        )
        return {}

    # Key normalization is handled by load_yaml_file, so keys in user_data should be normalized.
    # Validation logic below assumes normalized keys (lowercase_with_underscores)
    expected_top_level_keys = [
        "novel_concept",
        "protagonist",
        "plot_points",
        "setting",
        "world_details",
        "antagonist",
        "conflict",
        "other_key_characters",
    ]
    found_any_expected_key = False
    for key in expected_top_level_keys:
        if key in user_data and isinstance(user_data[key], dict) and user_data[key]:
            found_any_expected_key = True
            break
        elif (
            key == "plot_points"
            and key in user_data
            and isinstance(user_data[key], list)
            and user_data[key]
        ):
            found_any_expected_key = True
            break

    if not found_any_expected_key:
        logger.error(
            f"YAML '{yaml_file_path}' lacked expected sections after normalization: {user_data}"
        )
        return {}

    try:
        validated = UserStoryInputModel.model_validate(user_data)
    except ValidationError as exc:
        logger.error(
            "Validation error parsing user story YAML: %s",
            exc,
        )
        return {}

    logger.info(f"Loaded and validated user story data from '{yaml_file_path}'.")
    return validated


def _populate_agent_state_from_user_data(agent: Any, user_data: Dict[str, Any]):
    plot_outline: PlotOutlineData = (
        agent.plot_outline
        if hasattr(agent, "plot_outline") and agent.plot_outline
        else {}
    )
    character_profiles: Dict[str, Any] = (
        agent.character_profiles
        if hasattr(agent, "character_profiles") and agent.character_profiles
        else {}
    )
    world_building: WorldBuildingData = (
        agent.world_building
        if hasattr(agent, "world_building") and agent.world_building
        else {}
    )

    # Initialize world_building structure
    for cat_internal_key in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.values():
        world_building.setdefault(cat_internal_key, {})
    world_building["user_supplied_data"] = True  # Mark that user data was involved
    world_building["is_default"] = False
    world_building["source"] = "user_supplied_yaml"  # Updated source identifier

    nc = user_data.get(
        "novel_concept", {}
    )  # Assumes keys in user_data are already normalized by load_yaml_file
    plot_outline["title"] = _get_val_or_fill_in(nc, "title")
    plot_outline["genre"] = _get_val_or_fill_in(nc, "genre")
    plot_outline["theme"] = _get_val_or_fill_in(nc, "theme")
    plot_outline["logline"] = _get_val_or_fill_in(nc, "logline")

    prot_data = user_data.get("protagonist", {})
    plot_outline["protagonist_name"] = _get_val_or_fill_in(prot_data, "name")
    plot_outline["protagonist_description"] = _get_val_or_fill_in(
        prot_data, "description"
    )
    plot_outline["character_arc"] = _get_val_or_fill_in(prot_data, "character_arc")

    ant_data = user_data.get("antagonist", {})
    plot_outline["antagonist_name"] = _get_val_or_fill_in(ant_data, "name")
    plot_outline["antagonist_description"] = _get_val_or_fill_in(
        ant_data, "description"
    )
    plot_outline["antagonist_motivations"] = _get_val_or_fill_in(
        ant_data, "motivations"
    )

    # Get plot_elements data, defaulting to an empty dict if not present
    plot_elements_data = user_data.get("plot_elements", {})
    conflict_data = user_data.get(
        "conflict", {}
    )  # Still needed for climax_event_preview unless specified otherwise

    if plot_elements_data:
        logger.info(
            "Populating plot outline from 'plot_elements' section in user data."
        )
        plot_outline["inciting_incident"] = _get_val_or_fill_in(
            plot_elements_data, "inciting_incident"
        )
        # Read key_plot_points from plot_elements_data
        raw_plot_points = plot_elements_data.get("key_plot_points", [])
        plot_outline["conflict_summary"] = _get_val_or_fill_in(
            plot_elements_data, "central_conflict"
        )
        plot_outline["stakes"] = _get_val_or_fill_in(plot_elements_data, "stakes")
        # climax_event_preview continues to be read from conflict_data as its migration was not specified.
        # If it should also come from plot_elements, this would need adjustment.
        plot_outline["climax_event_preview"] = _get_val_or_fill_in(
            conflict_data, "climax_event_preview"
        )
    else:
        logger.info(
            "No 'plot_elements' section found or it is empty. Using fallback logic or default fill-ins."
        )
        # If plot_elements is the sole intended source for these, they become [Fill-in]
        plot_outline["inciting_incident"] = _get_val_or_fill_in(
            {}, "inciting_incident"
        )  # Results in [Fill-in]
        # Fallback to reading plot_points from the top-level user_data if plot_elements not present
        raw_plot_points = user_data.get("plot_points", [])
        plot_outline["conflict_summary"] = _get_val_or_fill_in(
            conflict_data,
            "summary",  # Fallback to old source if plot_elements not present
        )
        plot_outline["stakes"] = _get_val_or_fill_in(
            {}, "stakes"
        )  # Results in [Fill-in] as it's new to plot_elements
        plot_outline["climax_event_preview"] = _get_val_or_fill_in(
            conflict_data, "climax_event_preview"
        )

    if not isinstance(raw_plot_points, list):
        logger.warning(
            f"Plot points ('key_plot_points' or 'plot_points') parsed as non-list: {type(raw_plot_points)}. Defaulting to [Fill-in]."
            f"Plot points ('key_plot_points' or 'plot_points') parsed as non-list: {type(raw_plot_points)}. Defaulting to [Fill-in]."
        )
        plot_outline["plot_points"] = [
            config.MARKDOWN_FILL_IN_PLACEHOLDER
        ] * config.TARGET_PLOT_POINTS_INITIAL_GENERATION
    else:
        # Ensure each plot point is a string, handle if not.
        processed_plot_points = []
        for pp in raw_plot_points:
            if isinstance(pp, str):
                processed_plot_points.append(
                    pp.strip()
                    if pp.strip() or utils._is_fill_in(pp)
                    else config.MARKDOWN_FILL_IN_PLACEHOLDER
                )
            elif pp is None:  # Handle None items in the list
                processed_plot_points.append(config.MARKDOWN_FILL_IN_PLACEHOLDER)
            else:  # Coerce to string if other type, e.g. number
                processed_plot_points.append(
                    str(pp).strip()
                    if str(pp).strip()
                    else config.MARKDOWN_FILL_IN_PLACEHOLDER
                )

        plot_outline["plot_points"] = processed_plot_points

        # Pad with [Fill-in] placeholders up to TARGET_PLOT_POINTS_INITIAL_GENERATION
        # Ensure that we only pad if there are actual plot points or if the list is empty
        # and needs to be filled to the target length.
        # Also, ensure that if the list is shorter than target, but the last item is already a fill-in,
        # we don't add more fill-ins unless necessary to reach the target.

        # Only add fill-ins if the list is shorter than the target
        current_length = len(plot_outline["plot_points"])
        if current_length < config.TARGET_PLOT_POINTS_INITIAL_GENERATION:
            # If the list is not empty and the last element is not already a fill-in,
            # or if the list is empty, then pad.
            # This prevents adding fill-ins if the user explicitly provided some fill-ins at the end
            # but fewer than the target. The goal is to reach the target.
            if (
                current_length == 0
                or (
                    current_length > 0
                    and plot_outline["plot_points"][-1]
                    != config.MARKDOWN_FILL_IN_PLACEHOLDER
                )
                or sum(
                    1
                    for p in plot_outline["plot_points"]
                    if p == config.MARKDOWN_FILL_IN_PLACEHOLDER
                )
                < (config.TARGET_PLOT_POINTS_INITIAL_GENERATION - current_length)
            ):
                needed_fill_ins = (
                    config.TARGET_PLOT_POINTS_INITIAL_GENERATION - current_length
                )
                plot_outline["plot_points"].extend(
                    [config.MARKDOWN_FILL_IN_PLACEHOLDER] * needed_fill_ins
                )

        # Ensure the list does not exceed the target length if it somehow became too long before padding
        if (
            len(plot_outline["plot_points"])
            > config.TARGET_PLOT_POINTS_INITIAL_GENERATION
        ):
            plot_outline["plot_points"] = plot_outline["plot_points"][
                : config.TARGET_PLOT_POINTS_INITIAL_GENERATION
            ]

    # Process 'setting' section from user_data for world_building
    setting_data_md = user_data.get("setting", {})
    overview_desc_from_setting = _get_val_or_fill_in(
        setting_data_md, "primary_setting_description"
    )
    plot_outline["setting_description"] = (
        overview_desc_from_setting  # Also store in plot_outline
    )

    world_building.setdefault("_overview_", {})
    if not utils._is_fill_in(
        overview_desc_from_setting
    ):  # Only overwrite if user provided something
        world_building["_overview_"]["description"] = overview_desc_from_setting

    key_locations_md = setting_data_md.get(
        "key_locations", {}
    )  # This is usually like {"the_hourglass_curios": {"description": "..."}}
    if isinstance(key_locations_md, dict):
        world_building.setdefault("locations", {})
        for loc_name_norm_md, loc_details_md in key_locations_md.items():
            loc_name_display = loc_name_norm_md.replace("_", " ").title()
            if not utils._is_fill_in(loc_name_display) and isinstance(
                loc_details_md, dict
            ):
                agent_loc_details = world_building["locations"].setdefault(
                    loc_name_display, {"source": "user_supplied_markdown"}
                )
                for md_detail_key, md_detail_val in loc_details_md.items():
                    internal_detail_key = (
                        WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(
                            md_detail_key, md_detail_key
                        )
                    )
                    agent_loc_details[internal_detail_key] = md_detail_val

    # Process 'world_details' section from user_data
    wd_details_md = user_data.get("world_details", {})
    if isinstance(wd_details_md, dict):
        for main_wd_key_norm_md, main_wd_content_md in wd_details_md.items():
            if main_wd_key_norm_md == "unique_world_feature":
                target_category_internal = "systems"
                world_building.setdefault(target_category_internal, {})
                if isinstance(main_wd_content_md, dict):
                    # Assume "Unique World Feature" as the display name if not explicitly named in MD
                    feature_item_name_display = (
                        _get_val_or_fill_in(
                            main_wd_content_md,
                            "name",
                            default_is_fill_in=False,
                        )
                        or "Unique World Feature"
                    )
                    agent_item_details = world_building[
                        target_category_internal
                    ].setdefault(
                        feature_item_name_display,
                        {"source": "user_supplied_markdown"},
                    )
                    for (
                        md_detail_key,
                        md_detail_val,
                    ) in main_wd_content_md.items():
                        if md_detail_key == "name":
                            continue  # Already used for item name
                        internal_detail_key = (
                            WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(
                                md_detail_key, md_detail_key
                            )
                        )
                        agent_item_details[internal_detail_key] = md_detail_val

            elif main_wd_key_norm_md == "key_factions":
                target_category_internal = "factions"
                world_building.setdefault(target_category_internal, {})
                if isinstance(
                    main_wd_content_md, dict
                ):  # Content should be dict of faction_name: details
                    for (
                        item_name_norm_md,
                        item_details_md,
                    ) in main_wd_content_md.items():
                        item_name_display = item_name_norm_md.replace("_", " ").title()
                        if not utils._is_fill_in(item_name_display) and isinstance(
                            item_details_md, dict
                        ):
                            agent_item_details = world_building[
                                target_category_internal
                            ].setdefault(
                                item_name_display,
                                {"source": "user_supplied_markdown"},
                            )
                            for (
                                md_detail_key,
                                md_detail_val,
                            ) in item_details_md.items():
                                internal_detail_key = (
                                    WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(
                                        md_detail_key, md_detail_key
                                    )
                                )
                                agent_item_details[internal_detail_key] = md_detail_val

            elif main_wd_key_norm_md == "relevant_lore":
                target_category_internal = "lore"
                world_building.setdefault(target_category_internal, {})
                if isinstance(main_wd_content_md, dict):
                    for (
                        item_name_norm_md,
                        item_details_md,
                    ) in main_wd_content_md.items():
                        item_name_display = item_name_norm_md.replace("_", " ").title()
                        if not utils._is_fill_in(item_name_display) and isinstance(
                            item_details_md, dict
                        ):
                            agent_item_details = world_building[
                                target_category_internal
                            ].setdefault(
                                item_name_display,
                                {"source": "user_supplied_markdown"},
                            )
                            for (
                                md_detail_key,
                                md_detail_val,
                            ) in item_details_md.items():
                                internal_detail_key = (
                                    WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.get(
                                        md_detail_key, md_detail_key
                                    )
                                )
                                agent_item_details[internal_detail_key] = md_detail_val
            # Add other categories like "history", "society" similarly if they appear under world_details

    plot_outline["source"] = "user_supplied_yaml"  # Updated source identifier
    plot_outline["is_default"] = False
    agent.plot_outline = plot_outline

    # Ensure character_profiles is a dictionary of CharacterProfile instances
    if not isinstance(
        character_profiles, dict
    ):  # Should have been initialized earlier, but as a safeguard
        character_profiles = {}

    prot_name_val = plot_outline.get("protagonist_name")
    if (
        not utils._is_fill_in(prot_name_val) and prot_name_val
    ):  # Ensure prot_name_val is not empty
        if prot_name_val not in character_profiles or not isinstance(
            character_profiles[prot_name_val], CharacterProfile
        ):
            profile = CharacterProfile(name=prot_name_val)
            character_profiles[prot_name_val] = profile
        else:
            profile = character_profiles[prot_name_val]

        profile.description = plot_outline.get(
            "protagonist_description", config.MARKDOWN_FILL_IN_PLACEHOLDER
        )
        profile.traits = [
            t
            for t in prot_data.get("traits", [])
            if isinstance(t, str) and (t.strip() or utils._is_fill_in(t))
        ]
        profile.status = (
            _get_val_or_fill_in(prot_data, "initial_status") or "As described"
        )
        profile.relationships = prot_data.get("relationships", {})  # Direct attribute

        profile.updates["character_arc_summary"] = plot_outline.get(
            "character_arc", config.MARKDOWN_FILL_IN_PLACEHOLDER
        )
        profile.updates["role"] = "protagonist"
        profile.updates["source"] = "user_supplied_yaml"

    ant_name_val = plot_outline.get("antagonist_name")
    if (
        not utils._is_fill_in(ant_name_val) and ant_name_val and ant_data
    ):  # Ensure ant_name_val is not empty
        if ant_name_val not in character_profiles or not isinstance(
            character_profiles[ant_name_val], CharacterProfile
        ):
            ant_profile = CharacterProfile(name=ant_name_val)
            character_profiles[ant_name_val] = ant_profile
        else:
            ant_profile = character_profiles[ant_name_val]

        ant_profile.description = plot_outline.get(
            "antagonist_description", config.MARKDOWN_FILL_IN_PLACEHOLDER
        )
        ant_profile.traits = [
            t
            for t in ant_data.get("traits", [])
            if isinstance(t, str) and (t.strip() or utils._is_fill_in(t))
        ]
        ant_profile.status = "As described"  # Direct attribute
        ant_profile.relationships = ant_data.get(
            "relationships", {}
        )  # Direct attribute

        ant_profile.updates["motivations"] = plot_outline.get(
            "antagonist_motivations", config.MARKDOWN_FILL_IN_PLACEHOLDER
        )
        ant_profile.updates["role"] = "antagonist"
        ant_profile.updates["source"] = "user_supplied_yaml"

    other_chars_data = user_data.get("other_key_characters", {})
    if isinstance(other_chars_data, dict):
        for (
            char_name_other_normalized_yaml,
            char_detail_yaml,
        ) in other_chars_data.items():
            char_name_key = (
                char_name_other_normalized_yaml  # This is already normalized key
            )

            if utils._is_fill_in(char_name_key) or not isinstance(
                char_detail_yaml, dict
            ):
                continue

            if char_name_key not in character_profiles or not isinstance(
                character_profiles[char_name_key], CharacterProfile
            ):
                other_char_profile = CharacterProfile(name=char_name_key)
                character_profiles[char_name_key] = other_char_profile
            else:
                other_char_profile = character_profiles[char_name_key]

            # Set source first, can be overwritten by yaml_detail_key if 'source' is in there
            other_char_profile.updates["source"] = "user_supplied_yaml"

            for yaml_detail_key, yaml_detail_val in char_detail_yaml.items():
                if (
                    yaml_detail_key == "name"
                ):  # Name is set at construction, display_name could be an update field
                    other_char_profile.updates["display_name"] = (
                        yaml_detail_val  # Example if display name differs
                    )
                elif yaml_detail_key == "description" and hasattr(
                    other_char_profile, "description"
                ):
                    other_char_profile.description = yaml_detail_val
                elif yaml_detail_key == "traits" and hasattr(
                    other_char_profile, "traits"
                ):
                    other_char_profile.traits = (
                        [str(t).strip() for t in yaml_detail_val]
                        if isinstance(yaml_detail_val, list)
                        else [str(yaml_detail_val).strip()]
                    )
                elif yaml_detail_key == "status" and hasattr(
                    other_char_profile, "status"
                ):
                    other_char_profile.status = yaml_detail_val
                elif yaml_detail_key == "relationships" and hasattr(
                    other_char_profile, "relationships"
                ):
                    other_char_profile.relationships = (
                        yaml_detail_val if isinstance(yaml_detail_val, dict) else {}
                    )
                else:  # Fallback to updates dictionary for other fields like role, specific motivations, etc.
                    other_char_profile.updates[yaml_detail_key] = yaml_detail_val

            # Ensure 'role' is set, defaulting to 'other_key_character' if not provided
            if "role" not in other_char_profile.updates and not hasattr(
                other_char_profile, "role"
            ):
                other_char_profile.updates["role"] = "other_key_character"

    agent.character_profiles = character_profiles
    agent.world_building = world_building  # world_building was modified in place
    logger.info(
        "Agent state populated from user-supplied YAML data (preserving '[Fill-in]' markers)."
    )


async def generate_plot_outline_logic(
    agent: Any, default_protagonist_name: str, unhinged_mode: bool, **kwargs
) -> Tuple[PlotOutlineData, Optional[Dict[str, int]]]:
    logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")
    user_supplied_data = _load_user_supplied_data()  # Now returns UserStoryInputModel

    llm_was_called = False
    accumulated_usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if user_supplied_data is not None:
        logger.info("Processing user-supplied YAML data for initial setup.")
        _populate_agent_state_from_user_data(
            agent,
            user_supplied_data.model_dump(exclude_none=True),
        )
    elif user_supplied_data == {}:
        logger.warning("YAML file empty or not a dictionary. Using LLM or defaults.")
        agent.plot_outline = {}  # Ensure it's an empty dict, not None
    else:  # File not found
        logger.info(
            "No YAML file found. Plot outline will be generated by LLM or defaults."
        )
        agent.plot_outline = {}

    # Ensure agent.plot_outline is a dict before checking keys
    if not isinstance(agent.plot_outline, dict):  # This is line 491 in original
        agent.plot_outline = {}

    logger.info(
        f"NANA_INIT_SETUP: State of agent.plot_outline before base_elements_for_outline construction: "
        f"Genre='{agent.plot_outline.get('genre')}', "
        f"Theme='{agent.plot_outline.get('theme')}', "
        f"Setting='{agent.plot_outline.get('setting_description')}', "
        f"Protagonist='{agent.plot_outline.get('protagonist_name')}'"
    )

    critical_plot_fields_for_llm_check = [
        "title",
        "protagonist_name",
        "protagonist_description",
        "setting_description",
        "conflict_summary",
        "genre",
        "theme",
    ]
    needs_llm_for_core_plot = not agent.plot_outline or any(
        utils._is_fill_in(agent.plot_outline.get(field))
        for field in critical_plot_fields_for_llm_check
    )

    plot_points_list_from_agent = agent.plot_outline.get("plot_points", [])
    if not isinstance(plot_points_list_from_agent, list):
        actual_plot_points_count = 0
        needs_llm_for_core_plot = True
        logger.warning(
            f"Plot points from agent state is not a list ({type(plot_points_list_from_agent)}). Will regenerate."
        )
    else:
        actual_plot_points = [
            pp
            for pp in plot_points_list_from_agent
            if not utils._is_fill_in(pp) and isinstance(pp, str) and pp.strip()
        ]
        actual_plot_points_count = len(actual_plot_points)

    if actual_plot_points_count < config.TARGET_PLOT_POINTS_INITIAL_GENERATION:
        needs_llm_for_core_plot = True
        if agent.plot_outline and plot_points_list_from_agent:
            logger.info(
                f"Insufficient concrete plot points ({actual_plot_points_count} provided vs {config.TARGET_PLOT_POINTS_INITIAL_GENERATION} target). LLM will supplement."
            )

    if needs_llm_for_core_plot:
        llm_was_called = True
        logger.info(
            "LLM generation required for core plot outline elements or to supplement plot points."
        )

        context_from_user_input = (
            "\n**User-Provided Context (Respect these if not '[Fill-in]'):**\n"
        )
        has_user_context = False
        current_plot_outline_key_map_for_llm_prompt = {
            k: v for k, v in PLOT_OUTLINE_KEY_MAP.items()
        }

        if agent.plot_outline:  # Should be a dict now
            for (
                display_key_normalized,
                internal_key,
            ) in current_plot_outline_key_map_for_llm_prompt.items():
                user_val = agent.plot_outline.get(internal_key)
                display_key_title_case = display_key_normalized.replace(
                    "_", " "
                ).title()

                if user_val is not None:
                    if isinstance(user_val, list):
                        concrete_list_items = [
                            item
                            for item in user_val
                            if not utils._is_fill_in(item) and str(item).strip()
                        ]
                        fill_in_placeholders_in_list = [
                            item for item in user_val if utils._is_fill_in(item)
                        ]

                        if concrete_list_items:
                            context_from_user_input += f"  - {display_key_title_case}: {len(concrete_list_items)} item(s) provided. "  # Ensure str
                            has_user_context = True
                        if fill_in_placeholders_in_list:
                            context_from_user_input += (
                                f"Also includes {len(fill_in_placeholders_in_list)} "
                                f"'{config.MARKDOWN_FILL_IN_PLACEHOLDER}' items to complete.\n"
                            )
                            has_user_context = True
                        elif concrete_list_items:
                            context_from_user_input += "This list might need expansion if below target count.\n"
                    elif not utils._is_fill_in(user_val) and str(user_val).strip():
                        context_from_user_input += (
                            f"  - {display_key_title_case}: {str(user_val)}\n"
                        )
                        has_user_context = True
                    elif utils._is_fill_in(user_val):
                        context_from_user_input += (
                            f"  - {display_key_title_case}: {config.MARKDOWN_FILL_IN_PLACEHOLDER} "
                            "(needs generation)\n"
                        )
                        has_user_context = True
        if not has_user_context:
            context_from_user_input = (
                "\n**User-Provided Context:** No overriding preferences or fill-in "
                "requests; generate all fields creatively.\n"
            )

        target_num_plot_points = config.TARGET_PLOT_POINTS_INITIAL_GENERATION

        base_elements_for_outline: Dict[str, Any] = {}
        # Ensure agent.plot_outline is a dict for checks below
        current_agent_plot_outline = (
            agent.plot_outline if isinstance(agent.plot_outline, dict) else {}
        )
        in_pure_llm_scratch_mode = not user_supplied_data and (
            not current_agent_plot_outline
            or not any(current_agent_plot_outline.values())
        )

        prompt_core_elements_intro = ""
        if unhinged_mode and in_pure_llm_scratch_mode:
            base_elements_for_outline["genre"] = kwargs.get(
                "genre", random.choice(config.UNHINGED_GENRES)
            )
            base_elements_for_outline["theme"] = kwargs.get(
                "theme", random.choice(config.UNHINGED_THEMES)
            )
            base_elements_for_outline["setting_description"] = kwargs.get(
                "setting_archetype",
                random.choice(config.UNHINGED_SETTINGS_ARCHETYPES),
            )
            base_elements_for_outline["protagonist_name"] = default_protagonist_name
            base_elements_for_outline["source_hint"] = "unhinged_pure_llm"
            prompt_core_elements_intro = (
                "You are in UNHINGED mode. Generate a novel concept based on:\n"
                f"  - Genre: {base_elements_for_outline['genre']}\n"
                f"  - Theme: {base_elements_for_outline['theme']}\n"
                f"  - Setting Archetype: {base_elements_for_outline['setting_description']}\n"
            )
        else:
            base_elements_for_outline["genre"] = (
                current_agent_plot_outline.get("genre")
                if not utils._is_fill_in(current_agent_plot_outline.get("genre"))
                else config.CONFIGURED_GENRE
            )
            base_elements_for_outline["theme"] = (
                current_agent_plot_outline.get("theme")
                if not utils._is_fill_in(current_agent_plot_outline.get("theme"))
                else config.CONFIGURED_THEME
            )
            base_elements_for_outline["setting_description"] = (
                current_agent_plot_outline.get("setting_description")
                if not utils._is_fill_in(
                    current_agent_plot_outline.get("setting_description")
                )
                else config.CONFIGURED_SETTING_DESCRIPTION
            )
            base_elements_for_outline["protagonist_name"] = (
                current_agent_plot_outline.get("protagonist_name")
                if not utils._is_fill_in(
                    current_agent_plot_outline.get("protagonist_name")
                )
                else default_protagonist_name
            )
            base_elements_for_outline["source_hint"] = (
                "configured_or_user_yaml"  # Updated source hint
            )
            prompt_core_elements_intro = (
                "Generate a novel concept based on (or incorporating):\n"
                f"  - Genre: {base_elements_for_outline['genre']}\n"
                f"  - Theme: {base_elements_for_outline['theme']}\n"
                f"  - Initial Setting Idea: {base_elements_for_outline['setting_description']}\n"
                f"  - Protagonist Name (if known): {base_elements_for_outline['protagonist_name']}\n"
            )

            prompt_core_elements_intro = (
                "Generate a novel concept based on (or incorporating):\n"
                f"  - Genre: {base_elements_for_outline['genre']}\n"
                f"  - Theme: {base_elements_for_outline['theme']}\n"
                f"  - Initial Setting Idea: {base_elements_for_outline['setting_description']}\n"
                f"  - Protagonist Name (if known): {base_elements_for_outline['protagonist_name']}\n"
            )

        logger.info(
            f"NANA_INIT_SETUP: Base elements determined for LLM prompt: "
            f"Genre='{base_elements_for_outline.get('genre')}', "
            f"Theme='{base_elements_for_outline.get('theme')}', "
            f"Setting='{base_elements_for_outline.get('setting_description')}', "
            f"Protagonist='{base_elements_for_outline.get('protagonist_name')}', "
            f"SourceHint='{base_elements_for_outline.get('source_hint')}'"
        )
        logger.info(
            f"NANA_INIT_SETUP: Constructed prompt_core_elements_intro for LLM: {prompt_core_elements_intro}"
        )
        prompt_lines = []
        if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
            prompt_lines.append("/no_think")

        json_keys_str = ", ".join([f'"{k}"' for k in PLOT_OUTLINE_KEY_MAP.keys()])
        json_list_keys_str = ", ".join(
            [f'"{k}"' for k in PLOT_OUTLINE_LIST_INTERNAL_KEYS]
        )

        prompt_lines.extend(
            [
                "You are a creative assistant specializing in crafting compelling narrative structures for full novels.",
                "Your task is to generate or complete a plot outline based on the provided context.",
                "You MUST output a single, valid JSON object.",
                "",
                "**JSON Structure Requirements:**",
                f"1. The root must be a single JSON object with the following keys: {json_keys_str}.",
                f"2. The value for the '{json_list_keys_str}' key MUST be a JSON array of strings.",
                "3. All other keys should have JSON string values.",
                "4. Ensure you generate a complete narrative arc with approximately "
                f"{target_num_plot_points} distinct points in the 'plot_points' array.",
                "",
                "**Example of Expected JSON Output:**",
                "```json",
                "{",
                '  "title": "Chronoscape Drifters",',
                '  "genre": "Time-Travel Adventure",',
                '  "theme": "The immutability of fate vs. free will",',
                '  "protagonist_name": "Jax Xenobia",',
                '  "protagonist_description": "A cynical historian who discovers a faulty time-travel device.",',
                '  "character_arc": "From a detached observer of the past to an active participant willing to risk everything to change it.",',
                '  "logline": "A historian who stumbles upon a malfunctioning time machine must navigate the chaotic streams of history to prevent a temporal paradox from erasing existence, all while being hunted by a relentless temporal guardian.",',
                '  "setting_description": "A near-future Earth where history is a commercialized commodity, accessed through sanitized virtual reality simulations.",',
                '  "conflict_summary": "Jax vs. The Chronos Guard (an organization enforcing temporal purity) and the internal conflict of whether saving one life is worth jeopardizing the entire timeline.",',
                '  "inciting_incident": "During a routine simulation, the device glitches, stranding Jax in a pivotal, unrecorded historical event where he accidentally saves someone who was supposed to die.",',
                '  "antagonist_name": "Warden Kael",',
                '  "antagonist_description": "A stoic and powerful agent of the Chronos Guard, who sees any deviation from the established timeline as a cosmic threat.",',
                '  "antagonist_motivations": "A rigid belief that the established timeline, with all its tragedies, is the only one that avoids total annihilation. He has witnessed the horrors of a paradoxical timeline before.",',
                '  "climax_event_preview": "A final confrontation at the nexus of time where Jax must choose between restoring the original timeline (and sacrificing the person he saved) or creating a new, uncertain reality.",',
                '  "plot_points": [',
                '    "Jax discovers the time-travel glitch and saves a historical figure, creating a small paradox.",',
                '    "Warden Kael detects the anomaly and begins hunting Jax through time.",',
                '    "Jax learns about the Chronos Guard and the dangers of altering history from a reclusive temporal outcast.",',
                '    "A cat-and-mouse chase ensues across multiple historical periods, with Jax using his historical knowledge to evade Kael.",',
                '    "The consequences of the initial paradox begin to ripple, subtly changing Jax\'s present.",',
                '    "Jax finds a way to amplify his device, hoping to make a more significant, corrective change.",',
                '    "Kael reveals a personal tragedy that fuels his rigid adherence to the timeline.",',
                '    "Jax must gather allies from different time periods who have also been affected by temporal anomalies.",',
                '    "The allies formulate a risky plan to access the nexus of time, the control point for the timeline.",',
                '    "Kael anticipates their move and sets a trap at the nexus.",',
                '    "The final confrontation occurs, where Jax must make a choice with cosmic consequences.",',
                '    "Resolution: Jax either restores the timeline and lives with the moral cost, or shatters it, creating a new, unknown future for everyone."'
                "  ]",
                "}",
                "```",
                "",
                f"{prompt_core_elements_intro}",
                f"{context_from_user_input}",
                "Based on all the above, generate a complete and valid JSON object following the structure requirements.",
                "If user context provided a concrete value for a field, you MUST use it in the JSON.",
                f"If user context for a field is '{config.MARKDOWN_FILL_IN_PLACEHOLDER}' or missing, or if a list like 'plot_points' needs more items, generate that content creatively.",
                "Begin your single, valid JSON output now. Do NOT include any explanatory text before or after the JSON object.",
            ]
        )
        prompt = "\n".join(prompt_lines)

        logger.info(
            f"Calling LLM for plot outline generation/completion (to JSON), targeting ~{target_num_plot_points} plot points..."
        )
        cleaned_outline_text, usage_data = await llm_service.async_call_llm(
            model_name=config.INITIAL_SETUP_MODEL,
            prompt=prompt,
            temperature=config.Temperatures.INITIAL_SETUP,
            stream_to_disk=True,
            frequency_penalty=config.FREQUENCY_PENALTY_INITIAL_SETUP,
            presence_penalty=config.PRESENCE_PENALTY_INITIAL_SETUP,
            auto_clean_response=True,
        )
        if usage_data:
            accumulated_usage_data["prompt_tokens"] += usage_data.get(
                "prompt_tokens", 0
            )
            accumulated_usage_data["completion_tokens"] += usage_data.get(
                "completion_tokens", 0
            )
            accumulated_usage_data["total_tokens"] += usage_data.get("total_tokens", 0)

        parsed_llm_response = None
        try:
            parsed_llm_response = json.loads(cleaned_outline_text)
            if not isinstance(parsed_llm_response, dict):
                logger.error(
                    "LLM plot outline output was not a JSON object. Response: %s",
                    cleaned_outline_text,
                )
                parsed_llm_response = None
        except json.JSONDecodeError:
            logger.error(
                "Failed to decode JSON for plot outline. Response: %s",
                cleaned_outline_text,
            )
            parsed_llm_response = None

        # This is inside the if parsed_llm_response: block
        # And after it's confirmed to be a dict (or handled if not)
        if parsed_llm_response:  # Check if parsed_llm_response is not None
            if isinstance(parsed_llm_response, dict):
                logger.info(
                    f"NANA_INIT_SETUP: LLM parsed response for critical fields: "
                    f"Genre='{parsed_llm_response.get('genre')}', "
                    f"Theme='{parsed_llm_response.get('theme')}', "
                    f"Setting='{parsed_llm_response.get('setting_description')}'"
                )
            else:
                # This case might be if json.loads succeeded but returned not a dict, or if parsed_llm_response was None initially
                logger.info(
                    f"NANA_INIT_SETUP: LLM parsed_llm_response was not a dictionary after JSON parsing. Value: {parsed_llm_response}"
                )

        if not isinstance(agent.plot_outline, dict):
            agent.plot_outline = {}  # Ensure it's a dict

        if parsed_llm_response:
            for key, llm_value in parsed_llm_response.items():
                internal_key = PLOT_OUTLINE_KEY_MAP.get(
                    key, key
                )  # Normalize key from JSON
                existing_val = agent.plot_outline.get(internal_key)

                if internal_key == "plot_points":
                    user_pps_concrete = [
                        pp
                        for pp in (existing_val or [])
                        if isinstance(pp, str)
                        and not utils._is_fill_in(pp)
                        and pp.strip()
                    ]
                    llm_pps = (
                        [pp for pp in llm_value if isinstance(pp, str) and pp.strip()]
                        if isinstance(llm_value, list)
                        else []
                    )

                    final_pps = user_pps_concrete[:]
                    for llm_pp_item in llm_pps:
                        if len(final_pps) >= target_num_plot_points:
                            break
                        if not any(
                            llm_pp_item.lower() in user_pp.lower()
                            or user_pp.lower() in llm_pp_item.lower()
                            for user_pp in user_pps_concrete
                        ):
                            final_pps.append(llm_pp_item)

                    while len(final_pps) < target_num_plot_points:
                        final_pps.append(
                            f"{config.MARKDOWN_FILL_IN_PLACEHOLDER} - Additional plot point needed"
                        )
                    agent.plot_outline[internal_key] = final_pps[
                        :target_num_plot_points
                    ]
                elif (
                    utils._is_fill_in(existing_val)
                    or existing_val is None
                    or not str(existing_val).strip()
                ):
                    agent.plot_outline[internal_key] = llm_value
                elif (
                    isinstance(existing_val, str)
                    and not existing_val.strip()
                    and llm_value
                ):  # Existing was empty string
                    agent.plot_outline[internal_key] = llm_value

            for be_key in ["genre", "theme", "setting_description"]:
                if utils._is_fill_in(agent.plot_outline.get(be_key)):
                    llm_generated_val = parsed_llm_response.get(
                        be_key, base_elements_for_outline.get(be_key)
                    )
                    if llm_generated_val and not utils._is_fill_in(llm_generated_val):
                        agent.plot_outline[be_key] = llm_generated_val
                    elif base_elements_for_outline.get(
                        be_key
                    ) and not utils._is_fill_in(base_elements_for_outline.get(be_key)):
                        agent.plot_outline[be_key] = base_elements_for_outline.get(
                            be_key
                        )

            logger.info(
                f"NANA_INIT_SETUP: Final agent.plot_outline after LLM processing and default application for critical fields: "
                f"Genre='{agent.plot_outline.get('genre')}', "
                f"Theme='{agent.plot_outline.get('theme')}', "
                f"Setting='{agent.plot_outline.get('setting_description')}'"
            )

            agent.plot_outline.pop("is_default", None)
            if (
                not agent.plot_outline.get("source")
                or agent.plot_outline.get("source") == "default_fallback"
            ):
                agent.plot_outline["source"] = base_elements_for_outline.get(
                    "source_hint", "llm_generated_or_merged"
                )
            logger.info(
                f"LLM updated plot outline '{agent.plot_outline.get('title', 'N/A')}' "
                f"with {len(agent.plot_outline.get('plot_points', []))} points."
            )
        else:
            logger.error(
                "LLM failed to provide a parsable core plot outline. Falling back to default if agent state is still insufficient."
            )
            if not agent.plot_outline or any(
                utils._is_fill_in(agent.plot_outline.get(field))
                for field in critical_plot_fields_for_llm_check
            ):
                agent.plot_outline = _create_default_plot(
                    default_protagonist_name,
                    base_elements_for_outline,
                    unhinged_mode and in_pure_llm_scratch_mode,
                )

    # Ensure agent.plot_outline is a dict for final checks
    if not isinstance(agent.plot_outline, dict):
        agent.plot_outline = {}

    prot_name_from_outline = agent.plot_outline.get("protagonist_name")
    if (
        utils._is_fill_in(prot_name_from_outline)
        or not isinstance(prot_name_from_outline, str)
        or not prot_name_from_outline.strip()
    ):
        agent.plot_outline["protagonist_name"] = default_protagonist_name
        logger.warning(
            f"Protagonist name resolved to default: {default_protagonist_name}"
        )

    final_protagonist_name = agent.plot_outline["protagonist_name"]

    if not hasattr(agent, "character_profiles") or agent.character_profiles is None:
        agent.character_profiles = {}
    if not isinstance(agent.character_profiles, dict):
        agent.character_profiles = {}

    if final_protagonist_name not in agent.character_profiles:
        prot_profile_ref = CharacterProfile(name=final_protagonist_name)
        agent.character_profiles[final_protagonist_name] = prot_profile_ref
    else:
        prot_profile_ref = agent.character_profiles[final_protagonist_name]
        if not isinstance(prot_profile_ref, CharacterProfile):
            logger.warning(
                f"Fixing corrupted state for protagonist '{final_protagonist_name}': was dict, now CharacterProfile."
            )
            prot_profile_ref = CharacterProfile.from_dict(
                final_protagonist_name,
                prot_profile_ref if isinstance(prot_profile_ref, dict) else {},
            )
            agent.character_profiles[final_protagonist_name] = prot_profile_ref

    prot_profile_ref.updates["source"] = agent.plot_outline.get("source", "unknown")

    for key, plot_key in [
        ("description", "protagonist_description"),
        ("character_arc_summary", "character_arc"),
    ]:
        plot_val = agent.plot_outline.get(plot_key)
        if not utils._is_fill_in(plot_val):
            if key == "description":
                prot_profile_ref.description = plot_val
            else:
                prot_profile_ref.updates[key] = plot_val
        elif key == "description" and (
            not prot_profile_ref.description
            or utils._is_fill_in(prot_profile_ref.description)
        ):
            prot_profile_ref.description = (
                f"{config.MARKDOWN_FILL_IN_PLACEHOLDER} for {key}"
            )
        elif key != "description" and (
            key not in prot_profile_ref.updates
            or utils._is_fill_in(prot_profile_ref.updates.get(key))
        ):
            prot_profile_ref.updates[key] = (
                f"{config.MARKDOWN_FILL_IN_PLACEHOLDER} for {key}"
            )

    prot_profile_ref.updates["role"] = "protagonist"
    if (
        "traits" not in prot_profile_ref.to_dict()
        or not prot_profile_ref.traits
        or (
            isinstance(prot_profile_ref.traits, list)
            and all(utils._is_fill_in(t) for t in prot_profile_ref.traits)
        )
    ):
        prot_profile_ref.traits = [config.MARKDOWN_FILL_IN_PLACEHOLDER]
    if not prot_profile_ref.status or utils._is_fill_in(prot_profile_ref.status):
        prot_profile_ref.status = (
            config.MARKDOWN_FILL_IN_PLACEHOLDER
            if utils._is_fill_in(agent.plot_outline.get("protagonist_description"))
            else "As described in plot outline"
        )

    logger.info(
        f"Finalized character profile for protagonist '{final_protagonist_name}'."
    )

    if not hasattr(agent, "world_building") or not isinstance(
        agent.world_building, dict
    ):
        agent.world_building = {}

    overview_items_dict = agent.world_building.setdefault("_overview_", {})
    if "_overview_" not in overview_items_dict or not isinstance(
        overview_items_dict["_overview_"], WorldItem
    ):
        overview_item = WorldItem.from_dict("_overview_", "_overview_", {})
        overview_items_dict["_overview_"] = overview_item
    else:
        overview_item = overview_items_dict["_overview_"]
    overview_desc_val = agent.plot_outline.get(
        "setting_description", config.MARKDOWN_FILL_IN_PLACEHOLDER
    )
    overview_item.properties["description"] = overview_desc_val
    agent.world_building["source"] = agent.plot_outline.get("source", "unknown")

    return (
        agent.plot_outline,
        accumulated_usage_data if llm_was_called else None,
    )


async def generate_world_building_logic(
    agent: Any,
) -> Tuple[WorldBuildingData, Optional[Dict[str, int]]]:
    llm_was_called = False
    accumulated_usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    # Ensure agent.world_building is initialized as a dict
    if not hasattr(agent, "world_building") or not isinstance(
        agent.world_building, dict
    ):
        agent.world_building = {}

    needs_llm_for_world = False
    if not agent.world_building:  # If it's empty after init or previous steps
        agent.world_building = {
            "source": "llm_to_create",
            "_overview_": {},
        }  # Default source
        needs_llm_for_world = True
    elif (
        agent.world_building.get("source") == "user_supplied_yaml"
    ):  # Updated source check
        # Check if any [Fill-in] exists in user-supplied data
        for (
            category_internal,
            items_or_details_dict,
        ) in agent.world_building.items():
            if category_internal in [
                "is_default",
                "source",
                "user_supplied_data",
            ]:
                continue  # Skip helper keys
            if isinstance(items_or_details_dict, dict):
                if category_internal == "_overview_":
                    for (
                        detail_key,
                        detail_value,
                    ) in items_or_details_dict.items():
                        if utils._is_fill_in(detail_value):
                            needs_llm_for_world = True
                            break
                else:  # Itemized categories
                    for (
                        item_name,
                        item_detail_dict,
                    ) in items_or_details_dict.items():
                        if isinstance(item_detail_dict, dict):
                            for (
                                detail_key,
                                detail_value,
                            ) in item_detail_dict.items():
                                if detail_key == "source":
                                    continue
                                if utils._is_fill_in(detail_value):
                                    needs_llm_for_world = True
                                    break
                                if isinstance(detail_value, list) and any(
                                    utils._is_fill_in(li) for li in detail_value
                                ):
                                    needs_llm_for_world = True
                                    break
                            if needs_llm_for_world:
                                break  # Break from items_or_details_dict loop
                if needs_llm_for_world:
                    break  # Break from categories loop
        if not needs_llm_for_world:
            logger.info(
                "Skipping LLM world-building generation: Data was user-supplied from YAML and seems complete (no [Fill-in]s)."
            )
            return agent.world_building, None
    else:  # Source is not user_supplied_yaml, or it's empty, assume LLM is needed
        needs_llm_for_world = True

    # Ensure plot_outline exists and setting_description is usable
    if (
        not hasattr(agent, "plot_outline")
        or not isinstance(agent.plot_outline, dict)
        or utils._is_fill_in(agent.plot_outline.get("setting_description"))
    ):
        logger.warning(
            "Cannot generate detailed world-building if plot outline or its setting_description is missing or '[Fill-in]'."
        )
        if not agent.world_building.get("_overview_", {}).get(
            "description"
        ) or utils._is_fill_in(
            agent.world_building.get("_overview_", {}).get("description")
        ):
            # Set a very generic overview if nothing else is available
            agent.world_building.setdefault("_overview_", {})["description"] = (
                "A world to be detailed by the LLM."
            )
        if (
            agent.world_building.get("source") != "user_supplied_yaml"
        ):  # Avoid overwriting this if it was set
            agent.world_building["source"] = "llm_generated_default_context"
        needs_llm_for_world = True  # Force LLM if essential context is missing

    if not needs_llm_for_world:  # This check might be redundant if logic above correctly sets needs_llm_for_world
        logger.info("World building data seems complete, skipping LLM call.")
        return agent.world_building, None

    llm_was_called = True
    plot_title = agent.plot_outline.get("title", "Untitled Novel")
    plot_genre = agent.plot_outline.get("genre", "N/A")

    # Prepare world_setting_desc for the prompt, prioritizing agent.world_building._overview.description
    world_setting_desc = agent.world_building.get("_overview_", {}).get("description")
    if not world_setting_desc or utils._is_fill_in(world_setting_desc):
        world_setting_desc = agent.plot_outline.get(
            "setting_description"
        )  # Fallback to plot_outline
    if not world_setting_desc or utils._is_fill_in(world_setting_desc):
        world_setting_desc = (
            "A mysterious and detailed world waiting to be fleshed out by the LLM."
        )

    user_world_context_str = "\n**User-Provided World Context (Content from user_story_elements.yaml - Respect these concrete values. Complete any '[Fill-in]' fields creatively using Markdown):**\n"  # Updated filename
    temp_user_wb_for_prompt: List[str] = []
    # Only build this section if the source is indeed user_supplied_yaml and it's not empty
    if agent.world_building.get("source") == "user_supplied_yaml" and (
        agent.world_building.get("_overview_")
        or any(
            k not in ["_overview_", "is_default", "source", "user_supplied_data"]
            for k in agent.world_building.keys()
        )
    ):  # user_supplied_data might be redundant
        overview_data_for_prompt = agent.world_building.get("_overview_", {})
        if overview_data_for_prompt:  # Check if overview itself is not empty
            temp_user_wb_for_prompt.append("## Overview")
            for (
                detail_key_internal,
                detail_val,
            ) in overview_data_for_prompt.items():
                if detail_key_internal == "source":
                    continue
                detail_key_display_for_prompt = detail_key_internal.replace(
                    "_", " "
                ).capitalize()
                temp_user_wb_for_prompt.append(
                    f"**{detail_key_display_for_prompt}**: {detail_val}"
                )
            temp_user_wb_for_prompt.append("")

        for cat_internal, items_dict in agent.world_building.items():
            if cat_internal in [
                "_overview_",
                "is_default",
                "source",
                "user_supplied_data",
            ]:
                continue

            cat_display_for_prompt = cat_internal.replace("_", " ").capitalize()
            category_lines_for_prompt: List[str] = []  # Lines for current category

            if isinstance(items_dict, dict) and items_dict:
                category_lines_for_prompt.append(f"## {cat_display_for_prompt}")
                for item_display_name, details in items_dict.items():
                    if not isinstance(details, dict) or item_display_name.startswith(
                        "_"
                    ):
                        continue  # Skip internal markers like _is_fill_in

                    category_lines_for_prompt.append(f"### {item_display_name}")
                    item_has_details = False
                    for detail_key_internal, detail_val in details.items():
                        if detail_key_internal == "source":
                            continue
                        detail_key_display_for_prompt = detail_key_internal.replace(
                            "_", " "
                        ).capitalize()
                        if isinstance(detail_val, list):
                            if detail_val:  # Only add if list has items
                                category_lines_for_prompt.append(
                                    f"**{detail_key_display_for_prompt}**:"
                                )
                                for li_val in detail_val:
                                    category_lines_for_prompt.append(f"  - {li_val}")
                                item_has_details = True
                        elif detail_val is not None and (
                            isinstance(detail_val, bool) or str(detail_val).strip()
                        ):
                            category_lines_for_prompt.append(
                                f"**{detail_key_display_for_prompt}**: {detail_val}"
                            )
                            item_has_details = True
                    if item_has_details:
                        category_lines_for_prompt.append("")

                if len(category_lines_for_prompt) > 1:  # Header + some content
                    temp_user_wb_for_prompt.extend(category_lines_for_prompt)

    if temp_user_wb_for_prompt:  # If any user context was actually formatted
        user_world_context_str += "\n".join(temp_user_wb_for_prompt)
    else:
        user_world_context_str = "\n**User-Provided World Context:** No specific world preferences or fill-in requests were found from user input; generate all fields creatively using Markdown.\n"

    # Constants for JSON prompt construction
    category_keys_str = ", ".join(
        [f'"{k}"' for k in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.keys()]
    )
    detail_keys_for_prompt = sorted(
        [
            k
            for k in WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.keys()
            if k != "primary_setting_description"
        ]
    )
    detail_keys_str = ", ".join([f'"{k}"' for k in detail_keys_for_prompt])

    internal_to_markdown_detail_key_map = {
        v: k for k, v in WORLD_DETAIL_KEY_MAP_FROM_MARKDOWN_TO_INTERNAL.items()
    }
    list_detail_keys_for_prompt = sorted(
        [
            internal_to_markdown_detail_key_map[k]
            for k in WORLD_DETAIL_LIST_INTERNAL_KEYS
            if k in internal_to_markdown_detail_key_map
        ]
    )
    list_keys_str = ", ".join([f'"{k}"' for k in list_detail_keys_for_prompt])

    # Construct the new prompt for JSON output
    prompt_lines = []
    if config.ENABLE_LLM_NO_THINK_DIRECTIVE:
        prompt_lines.append("/no_think")

    prompt_lines.append(f"""You are an expert world-building assistant for novelists.
Your goal is to generate a comprehensive and consistent set of world-building details.
You MUST output a single, valid JSON object that encompasses all world-building details. Do NOT use Markdown.

**JSON Structure Requirements:**

1.  **Top-Level Object:** The root of the JSON should be a single object.
2.  **Categories:** The top-level object should have keys for world-building categories. The primary categories are: {category_keys_str}.
    *   For the "overview" category, the value should be a JSON object containing detail key-value pairs. For example: `{{"description": "A vast desert world...", "mood": "Harsh and mysterious"}}`.
    *   For all other categories (e.g., "locations", "factions", "systems"), the value should be a JSON object. In this object, each key is the unique name of an item (e.g., "The Sunken City", "The Mechanists Guild"), and its value is another JSON object containing the details for that item.
3.  **Detail Keys:** The objects for "overview" and for each item within other categories (like "locations", "factions") should use the following keys for their details, where appropriate: {detail_keys_str}.
    *   Example for a location item: `{{"description": "A hidden oasis...", "atmosphere": "Peaceful but eerie"}}`
    *   Example for a faction item: `{{"description": "A group of scholars...", "goals": ["Preserve knowledge", "Share discoveries"]}}`
4.  **Value Types for Details:**
    *   Most detail values should be JSON strings.
    *   The following detail keys MUST have values that are JSON arrays of strings: {list_keys_str}.
        *   Example: `"goals": ["Achieve peace", "Build a monument"]`
        *   If there's only one item for such a key, it should still be in an array: `"key_events": ["The Great Upheaval"]`
        *   If there are no items for such a key, use an empty array: `"rules": []`

**Example of Expected JSON Output:**
```json
{{
  "overview": {{
    "description": "A world of floating islands interconnected by ancient bridges.",
    "mood": "Mysterious and serene",
    "time_period": "Post-cataclysmic era"
  }},
  "locations": {{
    "Aerie Citadel": {{
      "description": "The main hub city, built on the largest cluster of islands. Features advanced, wind-powered technology.",
      "atmosphere": "Bustling with traders, artisans, and sky-ship pilots. A sense of cautious optimism prevails."
    }},
    "Whispering Chasm": {{
      "description": "A deep rift between islands, rumored to hold ancient secrets and dangers.",
      "atmosphere": "Eerie, filled with strange echoes and gusts of wind."
    }}
  }},
  "factions": {{
    "Sky Wardens": {{
      "description": "Dedicated guardians of the old skyways and the inter-island bridges. They are stern and traditional.",
      "goals": ["Protect the floating islands from falling", "Preserve the knowledge of the Ancients"],
      "structure": "Hierarchical, led by a council of Elders."
    }}
  }},
  "systems": {{
    "Aetherium Currents": {{
      "description": "Naturally occurring energy currents in the sky that are harnessed for power and travel.",
      "rules": ["Currents shift seasonally", "Over-harvesting can lead to local depletion and atmospheric instability"],
      "function": "Primary power source for island technology and sky-ship propulsion."
    }}
  }}
  // ... other categories like "lore", "history", "society" would follow a similar structure ...
}}
```

**Handling User-Provided Context:**
The following section, delimited by "--- USER WORLD CONTEXT START ---" and "--- USER WORLD CONTEXT END ---", contains any world details already provided by the user. This context is currently in a Markdown-like format.
*   You MUST incorporate any concrete values from this user context into the correct locations within the JSON structure you generate.
*   If the user context specifies `"{config.MARKDOWN_FILL_IN_PLACEHOLDER}"` for a field, or if a field is missing from the user context that you would normally generate, you should creatively generate that field's content according to the JSON structure defined above.
*   Interpret headers (e.g., `## Locations`, `### Item Name`) and key-value pairs (e.g., `**Description**: ...`) from the user context to understand where the information belongs.

**Novel Concept (from Plot Outline):**
  - Title: {plot_title}
  - Genre: {plot_genre}
  - Core Setting Idea: {world_setting_desc}

--- USER WORLD CONTEXT START ---
{user_world_context_str}
--- USER WORLD CONTEXT END ---

Begin your single, valid JSON output now. Do NOT include any explanatory text before or after the JSON object.
""")
    prompt = "\n".join(prompt_lines)

    logger.info(
        "Generating/completing initial world-building data (to JSON) via LLM..."
    )  # Updated log message
    (
        cleaned_world_text_json,
        usage_data,
    ) = await llm_service.async_call_llm(  # Renamed variable
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt,
        temperature=config.Temperatures.INITIAL_SETUP,
        stream_to_disk=True,
        frequency_penalty=config.FREQUENCY_PENALTY_INITIAL_SETUP,
        presence_penalty=config.PRESENCE_PENALTY_INITIAL_SETUP,
        auto_clean_response=True,
    )
    if usage_data:
        accumulated_usage_data["prompt_tokens"] += usage_data.get("prompt_tokens", 0)
        accumulated_usage_data["completion_tokens"] += usage_data.get(
            "completion_tokens", 0
        )
        accumulated_usage_data["total_tokens"] += usage_data.get("total_tokens", 0)

    logger.debug(
        f"Cleaned LLM world-building JSON output (len: {len(cleaned_world_text_json)}):\nSTART_OF_WB_JSON_TEXT\n{cleaned_world_text_json}\nEND_OF_WB_JSON_TEXT"
    )  # Updated log message

    logger.info("Attempting to parse LLM world-building output using JSON parser...")
    parsed_llm_json_response: Optional[Dict[str, Any]] = None
    try:
        parsed_llm_json_response = json.loads(cleaned_world_text_json)
        logger.debug(
            f"DIRECT Output of json.loads for LLM response: {json.dumps(parsed_llm_json_response, indent=2)}"
        )
    except json.JSONDecodeError as e:
        logger.error(
            f"CRITICAL: Failed to decode JSON from LLM output. Error: {e}. Raw text was: {cleaned_world_text_json}"
        )
        parsed_llm_json_response = None

    if parsed_llm_json_response and isinstance(parsed_llm_json_response, dict):
        agent.world_building["source"] = (
            "llm_generated_or_merged_json_style"  # Set source upon successful LLM parse
        )

        if not isinstance(agent.world_building, dict):
            agent.world_building = {"_overview_": {}}

        for (
            json_top_level_cat_key,
            json_cat_content,
        ) in parsed_llm_json_response.items():
            internal_cat_name = WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.get(
                json_top_level_cat_key, None
            )
            if internal_cat_name is None:
                logger.warning(
                    f"LLM JSON response contained unrecognized top-level key '{json_top_level_cat_key}'. This key will be IGNORED."
                )
                continue

            agent.world_building.setdefault(internal_cat_name, {})
            target_category_in_agent = agent.world_building[internal_cat_name]

            if not isinstance(json_cat_content, dict):
                logger.warning(
                    f"Content for category '{json_top_level_cat_key}' from LLM JSON is not a dictionary. Skipping. Content: {json_cat_content}"
                )
                continue

            if internal_cat_name == "_overview_":
                overview_item = WorldItem.from_dict(
                    internal_cat_name, "_overview_", json_cat_content
                )
                target_category_in_agent["_overview_"] = overview_item
            else:
                agent.world_building[internal_cat_name] = {}
                target_category_in_agent = agent.world_building[internal_cat_name]

                is_single_item_style = all(
                    not isinstance(v, dict) for v in json_cat_content.values()
                ) and bool(json_cat_content)

                items_to_process = {}
                if is_single_item_style:
                    logger.info(
                        f"Category '{json_top_level_cat_key}' from LLM appears to be a single-item category. Bundling its properties into one item."
                    )
                    item_name = json_top_level_cat_key.replace("_", " ").title()
                    items_to_process[item_name] = json_cat_content
                else:
                    items_to_process = json_cat_content

                for item_name, item_details in items_to_process.items():
                    if not isinstance(item_details, dict):
                        logger.warning(
                            f"Item '{item_name}' in category '{json_top_level_cat_key}' is not a dictionary. Skipping."
                        )
                        continue

                    world_item_instance = WorldItem.from_dict(
                        internal_cat_name, item_name, item_details
                    )
                    target_category_in_agent[item_name] = world_item_instance

        agent.world_building.pop("is_default", None)
        agent.world_building.pop("user_supplied_data", None)
        logger.info(
            "Successfully processed LLM-generated JSON world-building into agent state."
        )
    else:
        logger.error(
            "Failed to parse world-building JSON; using existing or default data."
        )
        if not agent.world_building.get("_overview_") or utils._is_fill_in(
            agent.world_building.get("_overview_", {}).get("description")
        ):
            default_wb_overview_desc = agent.plot_outline.get(
                "setting_description", "A default world setting."
            )
            if utils._is_fill_in(default_wb_overview_desc):
                default_wb_overview_desc = (
                    "A default world setting, to be detailed later."
                )
            if not isinstance(agent.world_building, dict):
                agent.world_building = {}
            overview_items_dict = agent.world_building.setdefault("_overview_", {})
            if "_overview_" not in overview_items_dict or not isinstance(
                overview_items_dict.get("_overview_"), WorldItem
            ):
                overview_items_dict["_overview_"] = WorldItem.from_dict(
                    "_overview_", "_overview_", {}
                )
            overview_items_dict["_overview_"].properties["description"] = (
                default_wb_overview_desc
            )

            if agent.world_building.get("source") != "user_supplied_yaml":
                agent.world_building["source"] = "default_fallback"
            if agent.world_building.get("source") == "default_fallback":
                agent.world_building["is_default"] = True

    if llm_was_called:
        current_source = agent.world_building.get("source", "")
        if not (current_source == "user_supplied_yaml" and not needs_llm_for_world):
            for cat_internal_key in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.values():
                agent.world_building.setdefault(cat_internal_key, {})

    return (
        agent.world_building,
        accumulated_usage_data if llm_was_called else None,
    )
