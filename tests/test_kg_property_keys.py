from utils import kg_property_keys as keys


def test_key_generation():
    assert keys.elaboration_key(2) == "elaboration_in_chapter_2"
    assert keys.development_key(3) == "development_in_chapter_3"
    assert keys.source_quality_key(1) == "source_quality_chapter_1"
    assert keys.added_key(5) == "added_in_chapter_5"
    assert keys.updated_key(6) == "updated_in_chapter_6"


def test_parse_elaboration_key():
    assert keys.parse_elaboration_key("elaboration_in_chapter_7") == 7
    assert keys.parse_elaboration_key("elaboration_in_chapter_bad") is None
    assert keys.parse_elaboration_key("other_key") is None
