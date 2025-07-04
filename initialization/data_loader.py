import structlog
from config import settings
from pydantic import ValidationError
from yaml_parser import load_yaml_file

from models.user_input_models import UserStoryInputModel, user_story_to_objects

from .models import CharacterProfile, PlotOutline, WorldBuilding

logger = structlog.get_logger(__name__)


def load_user_supplied_model() -> UserStoryInputModel | None:
    """Load user story YAML into a validated model."""
    data = load_yaml_file(settings.USER_STORY_ELEMENTS_FILE_PATH)
    if not data:
        return None
    try:
        return UserStoryInputModel(**data)
    except ValidationError:  # pragma: no cover
        logger.exception("Failed to parse user story YAML")
        return None


def convert_model_to_objects(
    model: UserStoryInputModel,
) -> tuple[PlotOutline, dict[str, CharacterProfile], WorldBuilding]:
    """Convert a validated model into internal objects."""
    plot_data, characters, world_items = user_story_to_objects(model)
    plot_outline = PlotOutline(**plot_data)
    world_building = WorldBuilding(data=world_items)
    return plot_outline, characters, world_building
