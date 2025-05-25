# data_access/__init__.py
# This file makes the data_access directory a Python package.
# You can also use it to expose a simpler API from this package if desired.

from .plot_queries import save_plot_outline_to_db, get_plot_outline_from_db
from .character_queries import save_character_profiles_to_db, get_character_profiles_from_db, get_character_info_for_snippet_from_db
from .world_queries import save_world_building_to_db, get_world_building_from_db, get_world_elements_for_snippet_from_db
from .chapter_queries import (
    load_chapter_count_from_db,
    save_chapter_data_to_db,
    get_chapter_data_from_db,
    get_embedding_from_db,
    find_similar_chapters_in_db,
    get_all_past_embeddings_from_db,
)
from .kg_queries import (
    add_kg_triple_to_db,
    query_kg_from_db,
    get_most_recent_value_from_db,
)

__all__ = [
    "save_plot_outline_to_db", "get_plot_outline_from_db",
    "save_character_profiles_to_db", "get_character_profiles_from_db", "get_character_info_for_snippet_from_db",
    "save_world_building_to_db", "get_world_building_from_db", "get_world_elements_for_snippet_from_db",
    "load_chapter_count_from_db", "save_chapter_data_to_db", "get_chapter_data_from_db",
    "get_embedding_from_db", "find_similar_chapters_in_db", "get_all_past_embeddings_from_db",
    "add_kg_triple_to_db", "query_kg_from_db", "get_most_recent_value_from_db",
]