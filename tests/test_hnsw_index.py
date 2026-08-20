"""Document-level HNSW and retrieval regression tests."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from src.build_faiss_index import add_embeddings, build_one, main, validate_args
from src.document_retrieval import (
    adaptive_image_candidates,
    build_doc_image_rows,
    exact_score_documents,
)
from src.document_schema import retrieval_manifest_hash
from src.hnsw_final_benchmark import exact_all_document_scores, recall_at_k, top_rows
from src.search_pipeline import SearchPipeline


def normalized(values: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype="float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


class HnswBuildTests(unittest.TestCase):
    def test_build_save_load_and_inner_product_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = normalized([[1, 0], [0, 1], [1, 1]])
            np.save(root / "vectors.npy", vectors)
            info = build_one(root / "vectors.npy", root / "vectors.index", 2, 8, 40, 3, 2)
            index = faiss.read_index(str(root / "vectors.index"))
            scores, ids = index.search(vectors[:2], 1)
            self.assertEqual(index.ntotal, 3)
            self.assertEqual(ids[:, 0].tolist(), [0, 1])
            self.assertTrue(np.allclose(scores[:, 0], 1.0, atol=1e-5))
            self.assertGreater(info["size_bytes"], 0)

    def test_rejects_non_normalized_vectors(self) -> None:
        values = np.ones((4, 3), dtype="float32")
        index = faiss.IndexHNSWFlat(3, 4, faiss.METRIC_INNER_PRODUCT)
        with self.assertRaisesRegex(ValueError, "不是 l2 归一化向量"):
            add_embeddings(index, values, chunk_size=2, desc="bad")

    def test_low_memory_builder_outputs_float32_hnsw_flat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = normalized([[1, 0], [0, 1], [1, 1], [-1, 0]])
            np.save(root / "vectors.npy", vectors)
            info = build_one(
                root / "vectors.npy",
                root / "vectors.index",
                2,
                8,
                40,
                4,
                2,
                low_memory_build=True,
            )
            index = faiss.read_index(str(root / "vectors.index"))
            scores, ids = index.search(vectors, 1)
            self.assertIsInstance(index, faiss.IndexHNSWFlat)
            self.assertIsInstance(faiss.downcast_index(index.storage), faiss.IndexFlat)
            self.assertEqual(ids[:, 0].tolist(), [0, 1, 2, 3])
            self.assertTrue(np.allclose(scores[:, 0], 1.0, atol=1e-5))
            self.assertEqual(info["graph_build_storage"], "sq_fp16")
            self.assertEqual(info["final_storage"], "flat_float32")

    def test_rejects_invalid_parameters(self) -> None:
        args = argparse.Namespace(
            chunk_size=1,
            hnsw_m=0,
            ef_construction=200,
            benchmark_sample_size=1,
            candidate_pool_size=100,
        )
        with self.assertRaisesRegex(ValueError, "hnsw_m"):
            validate_args(args)

    def test_builds_different_text_and_image_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = [
                {"doc_row_id": 0, "doc_id": "d0", "text": "a", "images": [{"image_path": "a.jpg"}]},
                {"doc_row_id": 1, "doc_id": "d1", "text": "b", "images": [{"image_path": "b.jpg"}, {"image_path": "c.jpg"}]},
            ]
            images = [
                {"image_row_id": 0, "image_path": "a.jpg", "doc_row_ids": [0]},
                {"image_row_id": 1, "image_path": "b.jpg", "doc_row_ids": [1]},
                {"image_row_id": 2, "image_path": "c.jpg", "doc_row_ids": [1]},
            ]
            contract = {"model_fingerprint": "m", "data_manifest_hash": "d", "dimension": 2, "normalization": "l2"}
            manifest_hash = retrieval_manifest_hash(documents, images)
            meta = {
                **contract,
                "schema_version": 2,
                "document_count": 2,
                "image_count": 3,
                "relation_count": 3,
                "retrieval_manifest_hash": manifest_hash,
                "documents": documents,
                "image_assets": images,
            }
            (root / "retrieval_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            (root / "extract_state.json").write_text(
                json.dumps({**contract, "schema_version": 2, "document_count": 2, "image_count": 3, "retrieval_manifest_hash": manifest_hash, "complete": True}),
                encoding="utf-8",
            )
            np.save(root / "text_embeddings.npy", normalized([[1, 0], [0, 1]]))
            np.save(root / "image_embeddings.npy", normalized([[1, 0], [0, 1], [1, 1]]))
            with patch.object(sys, "argv", ["build_faiss_index.py", "--embedding_dir", str(root), "--hnsw_m", "8", "--ef_construction", "40"]):
                self.assertEqual(main(), 0)
            self.assertEqual(faiss.read_index(str(root / "text.index")).ntotal, 2)
            self.assertEqual(faiss.read_index(str(root / "image.index")).ntotal, 3)


class DocumentRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            {"doc_row_id": 0, "doc_id": "d0", "text": "red", "images": [{"image_path": "red.jpg"}, {"image_path": "blue.jpg"}]},
            {"doc_row_id": 1, "doc_id": "d1", "text": "green", "images": [{"image_path": "green.jpg"}]},
            {"doc_row_id": 2, "doc_id": "d2", "text": "shared", "images": [{"image_path": "blue.jpg"}]},
        ]
        self.image_assets = [
            {"image_row_id": 0, "image_path": "red.jpg", "doc_row_ids": [0]},
            {"image_row_id": 1, "image_path": "blue.jpg", "doc_row_ids": [0, 2]},
            {"image_row_id": 2, "image_path": "green.jpg", "doc_row_ids": [1]},
        ]
        self.text = normalized([[1, 0], [0, 1], [1, 1]])
        self.images = normalized([[1, 0], [0, 1], [-1, 0]])
        self.doc_images = build_doc_image_rows(3, self.image_assets)

    def test_exact_scoring_returns_one_complete_document_and_uses_max_image(self) -> None:
        results = exact_score_documents(
            [0, 1, 2],
            text_query=np.asarray([[0, 1]], dtype="float32"),
            image_query=None,
            text_embeddings=self.text,
            image_embeddings=self.images,
            documents=self.documents,
            image_assets=self.image_assets,
            doc_image_rows=self.doc_images,
            alpha=0.0,
        )
        self.assertEqual(results[0]["doc_id"], "d0")
        self.assertEqual(results[0]["score_ti"], 1.0)
        self.assertEqual(results[0]["image_paths"], ["red.jpg", "blue.jpg"])
        self.assertEqual(results[0]["image_count"], 2)
        self.assertEqual(results[0]["matched_image_path"], "blue.jpg")
        self.assertEqual(len({row["doc_id"] for row in results}), 3)

    def test_adaptive_image_candidates_expands_shared_image_to_all_owners(self) -> None:
        index = faiss.IndexFlatIP(2)
        index.add(self.images)
        docs, matched = adaptive_image_candidates(
            index, np.asarray([[0, 1]], dtype="float32"), self.image_assets, 2, multiplier=1
        )
        self.assertTrue({0, 2}.issubset(docs))
        self.assertEqual(matched[0], 1)
        self.assertEqual(matched[2], 1)

    def test_exact_all_scores_matches_candidate_rescoring(self) -> None:
        relation_images = np.asarray([0, 1, 1, 2], dtype=np.int64)
        relation_docs = np.asarray([0, 0, 2, 1], dtype=np.int64)
        query = np.asarray([[0, 1]], dtype="float32")
        scores = exact_all_document_scores(
            text_query=query,
            image_query=None,
            text_embeddings=self.text,
            image_embeddings=self.images,
            relation_image_rows=relation_images,
            relation_doc_rows=relation_docs,
            alpha=0.5,
        )
        ranked = exact_score_documents(
            range(3),
            text_query=query,
            image_query=None,
            text_embeddings=self.text,
            image_embeddings=self.images,
            documents=self.documents,
            image_assets=self.image_assets,
            doc_image_rows=self.doc_images,
            alpha=0.5,
        )
        self.assertEqual(top_rows(scores, 3), [row["doc_row_id"] for row in ranked])

    def test_recall(self) -> None:
        self.assertEqual(recall_at_k([0, 1, 2], [0, 2, 3], 2), 0.5)


class PipelineIndexLoadingTests(unittest.TestCase):
    def test_load_hnsw_stays_on_cpu_and_applies_ef_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.faiss"
            index = faiss.IndexHNSWFlat(2, 8, faiss.METRIC_INNER_PRODUCT)
            index.add(normalized([[1, 0], [0, 1]]))
            faiss.write_index(index, str(path))
            pipeline = SearchPipeline.__new__(SearchPipeline)
            pipeline.hnsw_ef_search = 77
            pipeline._hnsw_cpu_notice_printed = False
            pipeline.gpu_resources = []
            loaded = pipeline.load_index(path, use_gpu=True)
            self.assertEqual(loaded.hnsw.efSearch, 77)
            self.assertEqual(pipeline.gpu_resources, [])


if __name__ == "__main__":
    unittest.main()
