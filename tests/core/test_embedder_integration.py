"""Integration tests for the embedding layer.

These tests exercise fastembed end-to-end and require downloading a model on
first use. They are marked ``slow`` and also automatically skipped when
network access isn't available.
"""

from __future__ import annotations

import pytest

from axon.core.embeddings.embedder import _DEFAULT_DIMENSIONS, embed_graph, embed_query
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphNode, NodeLabel


@pytest.mark.slow
class TestEmbeddingIntegration:
    @pytest.fixture(autouse=True)
    def _require_network(self) -> None:
        """Skip in environments without model download access."""
        import socket

        try:
            socket.getaddrinfo("huggingface.co", 443)
        except OSError:
            pytest.skip("Network not available to download embedding model")

    def test_embed_query_returns_correct_dimensions(self) -> None:
        result = embed_query("a function that sorts a list")
        assert result is not None
        assert len(result) == _DEFAULT_DIMENSIONS

    def test_semantic_similarity_ranking(self) -> None:
        """Verify that semantically similar queries produce similar vectors."""
        import numpy as np

        vec_sort = embed_query("sorting algorithm")
        vec_db = embed_query("database connection pool")
        vec_sort2 = embed_query("function that sorts items in a list")

        assert vec_sort is not None
        assert vec_db is not None
        assert vec_sort2 is not None

        # sort and sort2 should be more similar than sort and db
        sim_related = np.dot(vec_sort, vec_sort2)
        sim_unrelated = np.dot(vec_sort, vec_db)
        assert sim_related > sim_unrelated, (
            f"Expected related similarity ({sim_related:.4f}) > "
            f"unrelated similarity ({sim_unrelated:.4f})"
        )

    def test_embed_graph_roundtrip(self) -> None:
        graph = KnowledgeGraph()
        graph.add_node(
            GraphNode(
                id="function:src/sort.py:quicksort",
                label=NodeLabel.FUNCTION,
                name="quicksort",
                file_path="src/sort.py",
                signature="def quicksort(arr: list) -> list:",
            )
        )
        results = embed_graph(graph)
        assert len(results) == 1
        assert len(results[0].embedding) == _DEFAULT_DIMENSIONS
