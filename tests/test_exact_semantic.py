import math
import unittest

import numpy as np

from rag_textbook_qa.indexing.exact import ExactVectorIndex


class ExactVectorTests(unittest.TestCase):
    def test_all_scores_and_rankings_match_an_independent_brute_force_reference(self):
        generator = np.random.default_rng(41)
        vectors = generator.normal(size=(89, 37)).astype(np.float32)
        query = generator.normal(size=37)
        identifiers = [f"chunk-{i:03}" for i in generator.permutation(89)]
        source = vectors.copy()
        index = ExactVectorIndex(vectors, identifiers)
        query_norm = math.sqrt(math.fsum(float(v)**2 for v in query))
        reference = [math.fsum(float(v) * float(q) for v, q in zip(row, query, strict=True))
                     / (math.sqrt(math.fsum(float(v)**2 for v in row)) * query_norm)
                     for row in vectors]
        expected = sorted(range(len(vectors)), key=lambda i: (-reference[i], identifiers[i]))
        for top_k in (1, 5, 89, 100):
            with self.subTest(top_k=top_k):
                ranked = index.search([query], top_k)
                self.assertEqual([i for i, _ in ranked], expected[:top_k])
                np.testing.assert_allclose([score for _, score in ranked],
                                           [reference[i] for i in expected[:top_k]], rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(vectors, source)
        self.assertEqual(index.matrix.dtype, np.float32)
        self.assertFalse(index.matrix.flags.writeable)

    def test_ties_use_ids_independently_of_get_order(self):
        for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
            ids = ["z", "a", "m"]
            index = ExactVectorIndex([[1., 0.] for _ in order], [ids[i] for i in order])
            self.assertEqual([index.chunk_ids[i] for i, _ in index.search([[1., 0.]], 3)], ["a", "m", "z"])

    def test_invalid_cached_matrices_fail_instead_of_falling_back(self):
        for vectors, ids in (([[1., 0.]], ["a", "b"]), ([[float("nan"), 1.]], ["a"]),
                             ([[1., float("inf")]], ["a"]), ([[0., 0.]], ["a"]),
                             ([[1.], [2., 3.]], ["a", "b"]), ([[1., 0.], [1., 0.]], ["a", "a"]),
                             ([[1e100, 1.]], ["a"]), ([[1., 0.]], [])):
            with self.subTest(vectors=vectors, ids=ids), self.assertRaises(ValueError):
                ExactVectorIndex(vectors, ids)

    def test_invalid_query_shape_dimension_values_and_top_k_are_explicit_errors(self):
        index = ExactVectorIndex([[1., 0.]], ["a"])
        with self.assertRaisesRegex(ValueError, "维度"):
            index.search([[1., 0., 0.]], 1)
        for query in ([1., 0.], [[1., 0.], [0., 1.]], [[float("nan"), 1.]], [[0., 0.]]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                index.search(query, 1)
        for count in (0, -1, True):
            with self.subTest(top_k=count), self.assertRaises(ValueError):
                index.search([[1., 0.]], count)
        self.assertEqual(ExactVectorIndex([], []).search([[1., 0.]], 1), [])
