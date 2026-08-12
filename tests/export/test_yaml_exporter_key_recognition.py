"""Regression tests for YAML export key recognition (issue #953).

Both YAML exporters build their output from a fixed set of keys, each
defaulting to an empty collection. A mapping keyed by anything else therefore
serialised to a structurally valid file with every collection empty: records
dropped, no exception raised, and the progress log reporting a completed
export. The only way to notice was to open the file.

Two things are pinned here.

Recognition: ``ContextGraph.to_dict()`` emits 'nodes'/'edges', so exporting a
context graph -- the most direct path from this library's own graph type to
YAML -- silently produced an empty file, including in
``examples/capability_gap_context_graphs_example.py``. Those keys are now
accepted as aliases for 'entities'/'relationships', matching how
``Neo4jCSVExporter._normalize_graph`` already treats the two vocabularies.

Rejection: a non-empty mapping sharing no key with the recognised set raises
``ProcessingError`` rather than writing an empty file. An empty mapping is
still allowed, since a graph with no records is legitimate and carries
nothing that could be lost.
"""

import os
import tempfile
import unittest

import yaml

from semantica.context import ContextGraph
from semantica.export.methods import export_yaml
from semantica.export.yaml_exporter import (
    SemanticNetworkYAMLExporter,
    YAMLSchemaExporter,
)
from semantica.utils.exceptions import ProcessingError

# Mappings whose keys the exporter does not read. Each would previously have
# produced a valid-looking file with every collection empty.
UNRECOGNIZED = {
    "json_envelope": {"data": [{"id": "1"}]},
    "records_key": {"records": [{"id": "1"}]},
    "typo_entitys": {"entitys": [{"id": "1"}]},
    "unrelated": {"foo": "bar"},
    "numeric_keys": {1: [{"id": "1"}]},
}

# Payloads that must export, keyed by the canonical names or their aliases.
RECOGNIZED = {
    "canonical": {"entities": [{"id": "1"}], "relationships": [{"id": "r"}]},
    "aliases": {"nodes": [{"id": "1"}], "edges": [{"id": "r"}]},
    "entities_only": {"entities": [{"id": "1"}]},
    "nodes_only": {"nodes": [{"id": "1"}]},
    "triplets_only": {"triplets": [{"s": "a", "p": "b", "o": "c"}]},
    "metadata_only": {"metadata": {"source": "test"}},
}


class TestUnrecognizedMappingsAreRejected(unittest.TestCase):
    """A mapping that would export empty must fail instead."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _path(self, name="out.yaml"):
        return os.path.join(self.tmpdir, name)

    def test_fixture_tables_are_populated(self):
        """Guard against a vacuous suite: emptying a table must not pass."""
        self.assertGreaterEqual(len(UNRECOGNIZED), 5)
        self.assertGreaterEqual(len(RECOGNIZED), 6)

    def test_unrecognized_mapping_raises(self):
        for label, payload in UNRECOGNIZED.items():
            with self.subTest(case=label):
                with self.assertRaises(ProcessingError):
                    export_yaml(payload, self._path())

    def test_error_names_the_offending_and_recognized_keys(self):
        """The message must be actionable, not merely the right type."""
        with self.assertRaises(ProcessingError) as ctx:
            export_yaml({"data": [{"id": "1"}]}, self._path())
        message = str(ctx.exception)
        self.assertIn("data", message)
        self.assertIn("entities", message)
        self.assertIn("nodes", message)

    def test_no_file_is_written_when_rejected(self):
        path = self._path("rejected.yaml")
        with self.assertRaises(ProcessingError):
            export_yaml({"data": [{"id": "1"}]}, path)
        self.assertFalse(os.path.exists(path))

    def test_schema_exporter_rejects_unrecognized_mappings(self):
        for label, payload in UNRECOGNIZED.items():
            with self.subTest(case=label):
                with self.assertRaises(ProcessingError):
                    export_yaml(payload, self._path(), method="schema")

    def test_exporter_classes_reject_directly(self):
        """Validation lives in the exporters, not only the wrapper."""
        for label, payload in UNRECOGNIZED.items():
            with self.subTest(exporter="SemanticNetwork", case=label):
                with self.assertRaises(ProcessingError):
                    SemanticNetworkYAMLExporter().export_semantic_network(payload)
            with self.subTest(exporter="Schema", case=label):
                with self.assertRaises(ProcessingError):
                    YAMLSchemaExporter().export_ontology_schema(payload)


class TestRecognizedMappingsStillExport(unittest.TestCase):
    """Everything the exporter reads must keep working."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _path(self, name="out.yaml"):
        return os.path.join(self.tmpdir, name)

    def test_recognized_payloads_export(self):
        for label, payload in RECOGNIZED.items():
            with self.subTest(case=label):
                path = self._path(f"{label}.yaml")
                export_yaml(payload, path)
                self.assertTrue(os.path.exists(path))

    def test_empty_mapping_is_still_allowed(self):
        """Deliberate exception: an empty graph carries nothing to lose."""
        for method in ("semantic_network", "schema"):
            with self.subTest(method=method):
                path = self._path(f"empty_{method}.yaml")
                export_yaml({}, path, method=method)
                self.assertTrue(os.path.exists(path))

    def test_schema_keys_export(self):
        path = self._path("schema.yaml")
        export_yaml(
            {"classes": [{"name": "Thing"}], "properties": []}, path, method="schema"
        )
        loaded = yaml.safe_load(open(path, encoding="utf-8"))
        self.assertEqual(loaded["classes"], [{"name": "Thing"}])


class TestNodesEdgesAliases(unittest.TestCase):
    """'nodes'/'edges' map onto 'entities'/'relationships'."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _export_and_load(self, payload, name="out.yaml"):
        path = os.path.join(self.tmpdir, name)
        export_yaml(payload, path)
        return yaml.safe_load(open(path, encoding="utf-8"))

    def test_nodes_and_edges_populate_entities_and_relationships(self):
        loaded = self._export_and_load(
            {"nodes": [{"id": "n1"}], "edges": [{"id": "e1"}]}, "aliased.yaml"
        )
        self.assertEqual(loaded["entities"], [{"id": "n1"}])
        self.assertEqual(loaded["relationships"], [{"id": "e1"}])

    def test_canonical_keys_are_unaffected_by_alias_support(self):
        loaded = self._export_and_load(
            {"entities": [{"id": "x"}], "relationships": [{"id": "y"}]},
            "canonical.yaml",
        )
        self.assertEqual(loaded["entities"], [{"id": "x"}])
        self.assertEqual(loaded["relationships"], [{"id": "y"}])

    def test_nodes_takes_precedence_over_entities(self):
        """Mirrors Neo4jCSVExporter._normalize_graph's precedence order."""
        loaded = self._export_and_load(
            {"nodes": [{"id": "from_nodes"}], "entities": [{"id": "from_entities"}]},
            "precedence.yaml",
        )
        self.assertEqual(loaded["entities"], [{"id": "from_nodes"}])

    def test_context_graph_to_dict_round_trips(self):
        """The regression that motivated alias support.

        ContextGraph.to_dict() emits 'nodes'/'edges'; exporting it previously
        wrote a file with every collection empty. This is the path used by
        examples/capability_gap_context_graphs_example.py.
        """
        graph = ContextGraph(advanced_analytics=False)
        graph.add_node("alice", "person")
        graph.add_node("acme", "org")
        graph.add_edge("alice", "acme", "works_at")

        loaded = self._export_and_load(graph.to_dict(), "context_graph.yaml")

        self.assertEqual(len(loaded["entities"]), 2)
        self.assertEqual(len(loaded["relationships"]), 1)
        exported_ids = {entity["id"] for entity in loaded["entities"]}
        self.assertEqual(exported_ids, {"alice", "acme"})


if __name__ == "__main__":
    unittest.main()
