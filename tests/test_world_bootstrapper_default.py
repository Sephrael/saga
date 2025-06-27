from initialization.bootstrappers.world_bootstrapper import create_default_world


def test_create_default_world_excludes_metadata_in_data():
    wb = create_default_world()
    assert "is_default" not in wb.data
    assert "source" not in wb.data
    assert wb.is_default is True
    assert wb.source == "default_fallback"
