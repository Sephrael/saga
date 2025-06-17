from utils.helpers import _is_fill_in
import config


def test_is_fill_in_helper():
    assert not _is_fill_in("abc")
    assert _is_fill_in(config.FILL_IN)
