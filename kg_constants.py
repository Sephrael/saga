"""Constants used for property names in the knowledge graph."""

# Relationship and node property names
KG_REL_CHAPTER_ADDED = "chapter_added"
KG_NODE_CREATED_CHAPTER = "created_chapter"
KG_NODE_CHAPTER_UPDATED = "chapter_updated"
KG_IS_PROVISIONAL = "is_provisional"

# Set of known relationship types used in the knowledge graph. This helps
# normalize variations and informs downstream consumers of valid types.
KG_REL_TYPES = {
    "IS_A",
    "IS_PART_OF",
    "IS_LOCATED_IN",
    "IS_NEAR",
    "HAS_TRAIT",
    "HAS_GOAL",
    "HAS_RULE",
    "HAS_KEY_ELEMENT",
    "KNOWS",
    "IS_DEAD",
    "IS_REMEMBERED_AS",
    "WAS_FRIEND_OF",
    "RELATED_TO",
}
