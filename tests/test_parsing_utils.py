import unittest
from parsing_utils import parse_rdf_triples_with_rdflib
import logging
import sys


class TestRdfTripleParsing(unittest.TestCase):
    def test_parse_simple_turtle(self):
        turtle_input = r"""
        @prefix : <http://example.org/saga/> .
        @prefix char: <http://example.org/saga/Character/> .
        @prefix loc: <http://example.org/saga/Location/> .
        @prefix prop: <http://example.org/saga/prop/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

        char:Jax rdf:type :Character ;
                 rdfs:label "Jax" ;
                 prop:hasAlias "J.X." ;
                 prop:livesIn loc:Hourglass_Curios ;
                 prop:hasOccupation "Antique Dealer" .

        loc:Hourglass_Curios rdf:type :Location ;
                             rdfs:label "Hourglass Curios" ;
                             prop:description "A dusty shop" .

        :SomeEvent prop:involvedCharacter char:Jax ;
                   prop:involvedCharacter char:Lila .

        char:Lila rdf:type :Character ;
                  rdfs:label "Lila" .
        """

        expected_triples_count = 9

        parsed_triples = parse_rdf_triples_with_rdflib(
            turtle_input, rdf_format="turtle"
        )

        self.assertEqual(len(parsed_triples), expected_triples_count)

        jax_alias_triple = next(
            (
                t
                for t in parsed_triples
                if t["subject"]["name"] == "Jax" and t["predicate"] == "hasAlias"
            ),
            None,
        )
        self.assertIsNotNone(jax_alias_triple, "Jax hasAlias triple not found")
        if jax_alias_triple:
            self.assertEqual(jax_alias_triple["object_literal"], "J.X.")
            self.assertTrue(jax_alias_triple["is_literal_object"])
            self.assertEqual(
                jax_alias_triple["subject"]["type"],
                "Character",
                "Jax type should be Character from rdf:type",
            )

        jax_livesin_triple = next(
            (
                t
                for t in parsed_triples
                if t["subject"]["name"] == "Jax" and t["predicate"] == "livesIn"
            ),
            None,
        )
        self.assertIsNotNone(jax_livesin_triple, "Jax livesIn triple not found")
        if jax_livesin_triple:
            self.assertFalse(jax_livesin_triple["is_literal_object"])
            self.assertIsNotNone(jax_livesin_triple["object_entity"])
            if jax_livesin_triple["object_entity"]:
                self.assertEqual(
                    jax_livesin_triple["object_entity"]["name"], "Hourglass Curios"
                )
                self.assertEqual(
                    jax_livesin_triple["object_entity"]["type"], "Location"
                )

        jax_type_triple = next(
            (
                t
                for t in parsed_triples
                if t["subject"]["name"] == "Jax"
                and t["predicate"] == "type"
                and t["object_entity"]["name"] == "Character"
            ),
            None,
        )
        self.assertIsNotNone(jax_type_triple, "Jax rdf:type Character triple not found")

        lila_type_triple = next(
            (
                t
                for t in parsed_triples
                if t["subject"]["name"] == "Lila"
                and t["predicate"] == "type"
                and t["object_entity"]["name"] == "Character"
            ),
            None,
        )
        self.assertIsNotNone(
            lila_type_triple, "Lila rdf:type Character triple not found"
        )

    def test_empty_input(self):
        parsed_triples = parse_rdf_triples_with_rdflib("", rdf_format="turtle")
        self.assertEqual(len(parsed_triples), 0)

    def test_invalid_turtle(self):
        invalid_turtle = (
            r"@prefix : <http://example.org/saga/> . char:Jax prop:hasAlias 'J.X.'"
        )
        parsed_triples = parse_rdf_triples_with_rdflib(
            invalid_turtle, rdf_format="turtle"
        )
        self.assertEqual(len(parsed_triples), 0)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()
