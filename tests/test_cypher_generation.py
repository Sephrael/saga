from kg_maintainer import (
    CharacterProfile,
    WorldItem,
    generate_character_node_cypher,
    generate_world_element_node_cypher,
)


def test_generate_character_node_cypher():
    profile = CharacterProfile("Alice", description="Hero", traits=["brave"])
    stmts = generate_character_node_cypher(profile)
    assert any("MERGE (c:Entity" in s[0] for s in stmts)
    assert any("HAS_CHARACTER" in s[0] for s in stmts)


def test_generate_world_element_node_cypher():
    item = WorldItem("places_city", "Places", "City", properties={"description": "Metropolis"})
    stmts = generate_world_element_node_cypher(item)
    assert any("MERGE (we:Entity" in s[0] for s in stmts)
    merge_stmt = stmts[0]
    assert merge_stmt[1]["name"] == "City"
    assert any("CONTAINS_ELEMENT" in s[0] for s in stmts)


def test_generate_world_element_node_cypher_nested_props():
    item = WorldItem(
        "places_forest",
        "Places",
        "Echo Forest",
        properties={"history": {"echo_keepers": "Keepers"}},
    )
    stmts = generate_world_element_node_cypher(item)
    params = stmts[0][1]
    assert isinstance(params["props"]["history"], str)
