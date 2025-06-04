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


def test_generate_world_element_node_cypher():
    item = WorldItem("places_city", "Places", "City", properties={"description": "Metropolis"})
    stmt, params = generate_world_element_node_cypher(item)
    assert "MERGE (we:Entity" in stmt
    assert params["id"] == "places_city"
