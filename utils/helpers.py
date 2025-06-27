from config import settings


def _is_fill_in(value: object) -> bool:
    """Return True if ``value`` equals the fill-in placeholder."""
    return isinstance(value, str) and value.strip() == settings.FILL_IN
