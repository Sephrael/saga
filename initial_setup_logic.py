# initial_setup_logic.py
import json
import logging
import os
import random
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from pydantic import ValidationError

import config
import utils  # For _is_fill_in
from kg_maintainer.models import CharacterProfile, WorldItem
from llm_interface import llm_service
from story_models import UserStoryInputModel, user_story_to_objects
from yaml_parser import load_yaml_file

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from nana_orchestrator import NANA_Orchestrator

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
        return None

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
        return None

    try:
        validated = UserStoryInputModel.model_validate(user_data)
    except ValidationError as exc:
        logger.error(
            "Validation error parsing user story YAML: %s",
            exc,
        )
        return None

    logger.info(f"Loaded and validated user story data from '{yaml_file_path}'.")
    return validated


def _populate_state_from_user_data(
    user_data: Dict[str, Any],
) -> Tuple[PlotOutlineData, Dict[str, CharacterProfile], WorldBuildingData]:
    plot_outline: PlotOutlineData = {}
    character_profiles: Dict[str, CharacterProfile] = {}
    world_building: WorldBuildingData = {}

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

    prot_name_val = plot_outline.get("protagonist_name")
    if not utils._is_fill_in(prot_name_val) and prot_name_val:
        profile = CharacterProfile(name=prot_name_val)
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
        profile.relationships = prot_data.get("relationships", {})
        profile.updates["character_arc_summary"] = plot_outline.get(
            "character_arc", config.MARKDOWN_FILL_IN_PLACEHOLDER
        )
        profile.updates["role"] = "protagonist"
        profile.updates["source"] = "user_supplied_yaml"
        character_profiles[prot_name_val] = profile

    ant_name_val = plot_outline.get("antagonist_name")
    if not utils._is_fill_in(ant_name_val) and ant_name_val and ant_data:
        ant_profile = CharacterProfile(name=ant_name_val)
        ant_profile.description = plot_outline.get(
            "antagonist_description", config.MARKDOWN_FILL_IN_PLACEHOLDER
        )
        ant_profile.traits = [
            t
            for t in ant_data.get("traits", [])
            if isinstance(t, str) and (t.strip() or utils._is_fill_in(t))
        ]
        ant_profile.status = "As described"
        ant_profile.relationships = ant_data.get("relationships", {})
        ant_profile.updates["motivations"] = plot_outline.get(
            "antagonist_motivations", config.MARKDOWN_FILL_IN_PLACEHOLDER
        )
        ant_profile.updates["role"] = "antagonist"
        ant_profile.updates["source"] = "user_supplied_yaml"
        character_profiles[ant_name_val] = ant_profile

    other_chars_data = user_data.get("other_key_characters", {})
    if isinstance(other_chars_data, dict):
        for (
            char_name_other_normalized_yaml,
            char_detail_yaml,
        ) in other_chars_data.items():
            char_name_key = char_name_other_normalized_yaml
            if utils._is_fill_in(char_name_key) or not isinstance(
                char_detail_yaml, dict
            ):
                continue
            other_char_profile = CharacterProfile(name=char_name_key)
            other_char_profile.updates["source"] = "user_supplied_yaml"
            for yaml_detail_key, yaml_detail_val in char_detail_yaml.items():
                if yaml_detail_key == "description":
                    other_char_profile.description = yaml_detail_val
                elif yaml_detail_key == "traits":
                    other_char_profile.traits = (
                        [str(t).strip() for t in yaml_detail_val]
                        if isinstance(yaml_detail_val, list)
                        else [str(yaml_detail_val).strip()]
                    )
                elif yaml_detail_key == "status":
                    other_char_profile.status = yaml_detail_val
                elif yaml_detail_key == "relationships":
                    other_char_profile.relationships = (
                        yaml_detail_val if isinstance(yaml_detail_val, dict) else {}
                    )
                else:
                    other_char_profile.updates[yaml_detail_key] = yaml_detail_val
            if "role" not in other_char_profile.updates:
                other_char_profile.updates["role"] = "other_key_character"
            character_profiles[char_name_key] = other_char_profile

    logger.info(
        "Agent state populated from user-supplied YAML data (preserving '[Fill-in]' markers)."
    )
    return plot_outline, character_profiles, world_building


async def generate_plot_outline_logic(
    current_plot_outline: PlotOutlineData,
    current_character_profiles: Dict[str, CharacterProfile],
    default_protagonist_name: str,
    unhinged_mode: bool,
    orchestrator: Optional["NANA_Orchestrator"] = None,
    **kwargs,
) -> Tuple[PlotOutlineData, Dict[str, CharacterProfile], Optional[Dict[str, int]]]:
    logger.info(f"Generating plot outline. Unhinged mode: {unhinged_mode}")

    plot_outline = current_plot_outline.copy()
    character_profiles = current_character_profiles.copy()

    user_supplied_data = _load_user_supplied_data()

    llm_was_called = False
    accumulated_usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if user_supplied_data is not None:
        logger.info("Processing user-supplied YAML data for initial setup.")
        if orchestrator is not None:
            orchestrator.load_state_from_user_model(user_supplied_data)
            plot_outline = orchestrator.plot_outline.copy()
            character_profiles = orchestrator.character_profiles.copy()
        else:
            plot_outline, character_profiles, _ = user_story_to_objects(
                user_supplied_data
            )
    else:
        logger.info(
            "No YAML file found. Plot outline will be generated by LLM or defaults."
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
    needs_llm_for_core_plot = not plot_outline or any(
        utils._is_fill_in(plot_outline.get(field))
        for field in critical_plot_fields_for_llm_check
    )

    plot_points_list = plot_outline.get("plot_points", [])
    if not isinstance(plot_points_list, list):
        actual_plot_points_count = 0
        needs_llm_for_core_plot = True
        logger.warning(
            f"Plot points from state is not a list ({type(plot_points_list)}). Will regenerate."
        )
    else:
        actual_plot_points = [
            pp
            for pp in plot_points_list
            if not utils._is_fill_in(pp) and isinstance(pp, str) and pp.strip()
        ]
        actual_plot_points_count = len(actual_plot_points)

    if actual_plot_points_count < config.TARGET_PLOT_POINTS_INITIAL_GENERATION:
        needs_llm_for_core_plot = True
        if plot_outline and plot_points_list:
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
        if plot_outline:
            for display_key_normalized, internal_key in PLOT_OUTLINE_KEY_MAP.items():
                user_val = plot_outline.get(internal_key)
                display_key_title_case = display_key_normalized.replace(
                    "_", " "
                ).title()
                if user_val is not None and not utils._is_fill_in(user_val):
                    context_from_user_input += (
                        f"  - {display_key_title_case}: {user_val}\n"
                    )
                    has_user_context = True

        if not has_user_context:
            context_from_user_input = "\n**User-Provided Context:** No initial values. Generate all fields creatively.\n"

        target_num_plot_points = config.TARGET_PLOT_POINTS_INITIAL_GENERATION

        base_elements_for_outline: Dict[str, Any] = {}
        in_pure_llm_scratch_mode = not user_supplied_data and not any(
            plot_outline.values()
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
                "setting_archetype", random.choice(config.UNHINGED_SETTINGS_ARCHETYPES)
            )
            base_elements_for_outline["protagonist_name"] = default_protagonist_name
            base_elements_for_outline["source_hint"] = "unhinged_pure_llm"
            prompt_core_elements_intro = f"You are in UNHINGED mode. Generate a novel concept based on:\n- Genre: {base_elements_for_outline['genre']}\n- Theme: {base_elements_for_outline['theme']}\n- Setting Archetype: {base_elements_for_outline['setting_description']}\n"
        else:
            base_elements_for_outline["genre"] = (
                plot_outline.get("genre")
                if not utils._is_fill_in(plot_outline.get("genre"))
                else config.CONFIGURED_GENRE
            )
            base_elements_for_outline["theme"] = (
                plot_outline.get("theme")
                if not utils._is_fill_in(plot_outline.get("theme"))
                else config.CONFIGURED_THEME
            )
            base_elements_for_outline["setting_description"] = (
                plot_outline.get("setting_description")
                if not utils._is_fill_in(plot_outline.get("setting_description"))
                else config.CONFIGURED_SETTING_DESCRIPTION
            )
            base_elements_for_outline["protagonist_name"] = (
                plot_outline.get("protagonist_name")
                if not utils._is_fill_in(plot_outline.get("protagonist_name"))
                else default_protagonist_name
            )
            base_elements_for_outline["source_hint"] = "configured_or_user_yaml"
            prompt_core_elements_intro = f"Generate a novel concept based on (or incorporating):\n- Genre: {base_elements_for_outline['genre']}\n- Theme: {base_elements_for_outline['theme']}\n- Initial Setting Idea: {base_elements_for_outline['setting_description']}\n- Protagonist Name (if known): {base_elements_for_outline['protagonist_name']}\n"

        json_keys_str = ", ".join([f'"{k}"' for k in PLOT_OUTLINE_KEY_MAP.keys()])
        prompt_lines = [
            "/no_think" if config.ENABLE_LLM_NO_THINK_DIRECTIVE else "",
            "You are a creative assistant specializing in crafting compelling narrative structures for full novels.",
            "Your task is to generate or complete a plot outline based on the provided context.",
            "You MUST output a single, valid JSON object.",
            f"1. The root must be a single JSON object with the following keys: {json_keys_str}.",
            "2. The value for 'plot_points' MUST be a JSON array of strings.",
            "3. All other keys should have JSON string values.",
            f"4. Ensure you generate a complete narrative arc with approximately {target_num_plot_points} distinct points in the 'plot_points' array.",
            prompt_core_elements_intro,
            context_from_user_input,
            "Based on all the above, generate a complete and valid JSON object following the structure requirements.",
        ]
        prompt = "\n".join(filter(None, prompt_lines))

        cleaned_outline_text, usage_data = await llm_service.async_call_llm(
            model_name=config.INITIAL_SETUP_MODEL,
            prompt=prompt,
            temperature=config.Temperatures.INITIAL_SETUP,
            stream_to_disk=True,
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

        try:
            parsed_llm_response = json.loads(cleaned_outline_text)
            if isinstance(parsed_llm_response, dict):
                plot_outline.update(parsed_llm_response)
                plot_outline["source"] = base_elements_for_outline.get(
                    "source_hint", "llm_generated_or_merged"
                )
                logger.info(
                    f"LLM successfully generated plot outline '{plot_outline.get('title', 'N/A')}' with {len(plot_outline.get('plot_points', []))} points."
                )
            else:
                raise json.JSONDecodeError(
                    "LLM output was not a JSON object.", cleaned_outline_text, 0
                )
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode or validate LLM JSON for plot outline: {e}. Falling back to default."
            )
            plot_outline = _create_default_plot(
                default_protagonist_name,
                base_elements_for_outline,
                unhinged_mode and in_pure_llm_scratch_mode,
            )

    final_protagonist_name = plot_outline.get(
        "protagonist_name", default_protagonist_name
    )
    if final_protagonist_name not in character_profiles:
        character_profiles[final_protagonist_name] = CharacterProfile(
            name=final_protagonist_name
        )

    prot_profile_ref = character_profiles[final_protagonist_name]
    prot_profile_ref.description = plot_outline.get("protagonist_description", "")
    prot_profile_ref.updates["character_arc_summary"] = plot_outline.get(
        "character_arc", ""
    )
    prot_profile_ref.updates["role"] = "protagonist"
    prot_profile_ref.updates["source"] = plot_outline.get("source", "unknown")

    logger.info(
        f"Finalized character profile for protagonist '{final_protagonist_name}'."
    )

    return (
        plot_outline,
        character_profiles,
        accumulated_usage_data if llm_was_called else None,
    )


async def generate_world_building_logic(
    current_world_building: WorldBuildingData,
    current_plot_outline: PlotOutlineData,
) -> Tuple[WorldBuildingData, Optional[Dict[str, int]]]:
    world_building = current_world_building.copy()
    plot_outline = current_plot_outline.copy()

    llm_was_called = False
    accumulated_usage_data: Dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    needs_llm_for_world = False
    if not world_building:
        needs_llm_for_world = True
    else:
        for category_internal, items_or_details in world_building.items():
            if category_internal in ["is_default", "source", "user_supplied_data"]:
                continue
            if isinstance(items_or_details, dict):
                if any(
                    utils._is_fill_in(v)
                    for v in items_or_details.values()
                    if not isinstance(v, (dict, list))
                ):
                    needs_llm_for_world = True
                    break
                for item_detail in items_or_details.values():
                    if isinstance(item_detail, dict) and any(
                        utils._is_fill_in(v) for v in item_detail.values()
                    ):
                        needs_llm_for_world = True
                        break
            if needs_llm_for_world:
                break

    if not needs_llm_for_world:
        logger.info("World building data seems complete, skipping LLM call.")
        return world_building, None

    llm_was_called = True
    plot_title = plot_outline.get("title", "Untitled Novel")
    plot_genre = plot_outline.get("genre", "N/A")
    world_setting_desc = plot_outline.get("setting_description", "A mysterious world.")

    world_json_keys = ", ".join(
        [f'"{k}"' for k in WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.keys()]
    )
    prompt_lines = [
        "/no_think" if config.ENABLE_LLM_NO_THINK_DIRECTIVE else "",
        (
            "You are a world-building assistant tasked with generating a rich "
            "setting for the upcoming novel."
        ),
        (
            f"Title: {plot_title} | Genre: {plot_genre}. The core setting "
            f"description is: {world_setting_desc}"
        ),
        (f"Return a single JSON object using these top-level keys: {world_json_keys}."),
        (
            "The 'overview' section should summarize the world and mood. Other "
            "sections should be objects keyed by item names with descriptive "
            "fields. Use lists for multi-value attributes like goals or rules."
        ),
        "Output only the JSON object with no extra commentary.",
    ]
    prompt = "\n".join(filter(None, prompt_lines))

    logger.info(
        "Generating/completing initial world-building data (to JSON) via LLM..."
    )
    cleaned_world_text_json, usage_data = await llm_service.async_call_llm(
        model_name=config.INITIAL_SETUP_MODEL,
        prompt=prompt,
        temperature=config.Temperatures.INITIAL_SETUP,
        auto_clean_response=True,
    )
    if usage_data:
        accumulated_usage_data.update(
            {k: accumulated_usage_data.get(k, 0) + v for k, v in usage_data.items()}
        )

    try:
        parsed_llm_json_response = json.loads(cleaned_world_text_json)
        if isinstance(parsed_llm_json_response, dict):
            world_building = {"source": "llm_generated_or_merged_json_style"}
            for cat_key, cat_content in parsed_llm_json_response.items():
                internal_cat_name = WORLD_CATEGORY_MAP_NORMALIZED_TO_INTERNAL.get(
                    cat_key
                )
                if internal_cat_name:
                    if internal_cat_name == "_overview_":
                        world_building[internal_cat_name] = {
                            "_overview_": WorldItem.from_dict(
                                internal_cat_name, "_overview_", cat_content
                            )
                        }
                    else:
                        world_building[internal_cat_name] = {
                            item_name: WorldItem.from_dict(
                                internal_cat_name, item_name, item_details
                            )
                            for item_name, item_details in cat_content.items()
                        }
            logger.info(
                "Successfully processed LLM-generated JSON world-building into state."
            )
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to decode world-building JSON: {e}. Using existing/default data."
        )
        world_building.setdefault("source", "default_fallback")

    return world_building, accumulated_usage_data if llm_was_called else None
