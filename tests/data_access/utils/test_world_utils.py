# tests/data_access/utils/test_world_utils.py

import copy
import unittest
from unittest.mock import patch

import kg_constants as kg_keys  # For testing functions that use these constants
from config import settings  # For default chapter num etc.
from data_access.utils import world_utils
from kg_maintainer.models import (
    WorldItem,  # Assuming WorldItem is used for get_world_item_by_name_from_data
)


class TestWorldUtils(unittest.TestCase):
    def setUp(self):
        # Clear cache before each test to ensure isolation
        world_utils.clear_world_name_to_id_cache()

    def test_build_world_element_id(self):
        details_with_id = {"id": "existing_id_123"}
        self.assertEqual(
            world_utils.build_world_element_id("cat", "name", details_with_id),
            "existing_id_123",
        )

        details_no_id = {"description": "some desc"}
        self.assertEqual(
            world_utils.build_world_element_id(
                "Location", "Central Park", details_no_id
            ),
            "location_central_park",
        )
        self.assertEqual(
            world_utils.build_world_element_id(
                "Magic System", "Elemental Weaving", details_no_id
            ),
            "magic_system_elemental_weaving",
        )

        # Test with empty/None values
        self.assertEqual(
            world_utils.build_world_element_id("", "Item Name", details_no_id),
            "unknown_category_item_name",
        )
        self.assertEqual(
            world_utils.build_world_element_id("Category", "", details_no_id),
            "category_unknown_name",
        )
        self.assertEqual(
            world_utils.build_world_element_id("", "", details_no_id),
            "unknown_category_unknown_name",
        )

    def test_world_name_to_id_cache_management(self):
        self.assertIsNone(world_utils.resolve_world_name_from_cache("Test Item"))

        world_utils.update_world_name_to_id_cache("Test Item", "test_item_id_1")
        self.assertEqual(
            world_utils.resolve_world_name_from_cache("Test Item"), "test_item_id_1"
        )
        self.assertEqual(
            world_utils.resolve_world_name_from_cache("test item"), "test_item_id_1"
        )  # Normalized lookup

        world_utils.update_world_name_to_id_cache("Another Item", "another_item_id_2")
        self.assertEqual(
            world_utils.resolve_world_name_from_cache("Another Item"),
            "another_item_id_2",
        )

        world_utils.clear_world_name_to_id_cache()
        self.assertIsNone(world_utils.resolve_world_name_from_cache("Test Item"))
        self.assertIsNone(world_utils.resolve_world_name_from_cache("Another Item"))

    def test_get_world_item_by_name_from_data(self):
        # Setup mock world_data and cache
        item1 = WorldItem(
            id="item_id_1", name="First Item", category="TestCat", description="Desc1"
        )
        item2 = WorldItem(
            id="item_id_2", name="Second Item", category="TestCat", description="Desc2"
        )
        world_data_mock = {
            "TestCat": {"First Item": item1, "Second Item": item2},
            "_overview_": {},  # Ensure overview doesn't break it
        }
        world_utils.update_world_name_to_id_cache("First Item", "item_id_1")
        world_utils.update_world_name_to_id_cache("Second Item", "item_id_2")

        self.assertEqual(
            world_utils.get_world_item_by_name_from_data(world_data_mock, "First Item"),
            item1,
        )
        self.assertEqual(
            world_utils.get_world_item_by_name_from_data(
                world_data_mock, "second item"
            ),
            item2,
        )  # Normalized
        self.assertIsNone(
            world_utils.get_world_item_by_name_from_data(
                world_data_mock, "NonExistent Item"
            )
        )

        # Test with empty world_data
        self.assertIsNone(
            world_utils.get_world_item_by_name_from_data({}, "First Item")
        )

    def test_process_elaborations_for_item(self):
        item_details = {}
        elaborations = [
            {"chapter": 1, "summary": "Elab ch1", "prov": False},
            {"chapter": 2, "summary": "Elab ch2 provisional", "prov": True},
            {"chapter": 3, "summary": "Elab ch3", "prov": False},
        ]

        # No chapter limit
        count = world_utils.process_elaborations_for_item(
            elaborations, None, item_details
        )
        self.assertEqual(count, 3)
        self.assertEqual(item_details[kg_keys.elaboration_key(1)], "Elab ch1")
        self.assertEqual(
            item_details[kg_keys.elaboration_key(2)], "Elab ch2 provisional"
        )
        self.assertIn(kg_keys.source_quality_key(2), item_details)
        self.assertEqual(
            item_details[kg_keys.source_quality_key(2)],
            "provisional_from_unrevised_draft",
        )
        self.assertNotIn(kg_keys.source_quality_key(1), item_details)

        # With chapter limit
        item_details_limited = {}
        count_limited = world_utils.process_elaborations_for_item(
            elaborations, 2, item_details_limited
        )
        self.assertEqual(count_limited, 2)
        self.assertIn(kg_keys.elaboration_key(1), item_details_limited)
        self.assertIn(kg_keys.elaboration_key(2), item_details_limited)
        self.assertNotIn(kg_keys.elaboration_key(3), item_details_limited)

    def test_extract_core_world_element_fields(self):
        node_valid = {"id": "id1", "name": "name1", "category": "cat1"}
        self.assertEqual(
            world_utils.extract_core_world_element_fields(node_valid),
            ("cat1", "name1", "id1"),
        )

        node_missing_name = {"id": "id2", "category": "cat2"}
        with patch.object(world_utils.logger, "debug") as mock_log_debug:
            self.assertEqual(
                world_utils.extract_core_world_element_fields(node_missing_name),
                (None, None, None),
            )
            mock_log_debug.assert_called_once()

        node_empty = {}
        self.assertEqual(
            world_utils.extract_core_world_element_fields(node_empty),
            (None, None, None),
        )

    def test_initialize_item_detail_dict_from_node(self):
        node = {
            "id": "item_id_1",
            "name": "Test Item",
            "category": "TestCat",
            kg_keys.KG_NODE_CREATED_CHAPTER: 5,
            kg_keys.KG_IS_PROVISIONAL: True,
            "description": "A test item.",
            "created_ts": 12345,
            "updated_ts": 67890,
        }
        expected_details = {
            "id": "item_id_1",
            "name": "Test Item",
            "category": "TestCat",
            "created_chapter": 5,
            kg_keys.added_key(5): True,
            "is_provisional": True,
            kg_keys.source_quality_key(5): "provisional_from_unrevised_draft",
            "description": "A test item.",
        }
        details, created_chap = world_utils.initialize_item_detail_dict_from_node(
            copy.deepcopy(node)
        )
        self.assertEqual(created_chap, 5)
        self.assertDictEqual(details, expected_details)

        # Test with default chapter and non-provisional
        node_defaults = {"id": "item_id_2", "name": "Default Item", "category": "Cat"}
        expected_details_defaults = {
            "id": "item_id_2",
            "name": "Default Item",
            "category": "Cat",
            "created_chapter": settings.KG_PREPOPULATION_CHAPTER_NUM,
            kg_keys.added_key(settings.KG_PREPOPULATION_CHAPTER_NUM): True,
            "is_provisional": False,
            # No source_quality key if not provisional
        }
        details_def, created_chap_def = (
            world_utils.initialize_item_detail_dict_from_node(
                copy.deepcopy(node_defaults)
            )
        )
        self.assertEqual(created_chap_def, settings.KG_PREPOPULATION_CHAPTER_NUM)
        self.assertDictEqual(details_def, expected_details_defaults)

    def test_populate_list_attributes_for_item(self):
        record = {
            "goals": [
                "Goal 2",
                "Goal 1",
                None,
                "Goal 3",
            ],  # Include None to test filtering
            "rules": ["Rule A"],
            "key_elements": [],  # Empty list
            "traits": None,  # Missing attribute
        }
        item_details = {}
        world_utils.populate_list_attributes_for_item(record, item_details)

        self.assertEqual(
            item_details["goals"], ["Goal 1", "Goal 2", "Goal 3"]
        )  # Sorted and None filtered
        self.assertEqual(item_details["rules"], ["Rule A"])
        self.assertEqual(item_details["key_elements"], [])
        self.assertEqual(
            item_details["traits"], []
        )  # Should default to empty list if attribute missing

    def test_should_include_world_item(self):
        # No chapter limit
        self.assertTrue(
            world_utils.should_include_world_item(1, 0, None, "item", "id1")
        )

        # Created within limit
        self.assertTrue(world_utils.should_include_world_item(1, 0, 5, "item", "id1"))
        self.assertTrue(world_utils.should_include_world_item(5, 0, 5, "item", "id1"))

        # Created after limit, no relevant elaborations
        with patch.object(world_utils.logger, "debug") as mock_log_debug:
            self.assertFalse(
                world_utils.should_include_world_item(6, 0, 5, "item", "id1")
            )
            mock_log_debug.assert_called_once()

        # Created after limit, but has relevant elaborations
        self.assertTrue(world_utils.should_include_world_item(6, 1, 5, "item", "id1"))


if __name__ == "__main__":
    unittest.main()
