"""HNSW 构建和加载的轻量回归测试。"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.build_faiss_index import add_embeddings, build_one, main, validate_args
from src.hnsw_final_benchmark import (
    FIXED_EF_SEARCH,
    final_ranked_keys,
    recall_at_k,
)
from src.model_fingerprint import MetadataMismatchError
from src.search_pipeline import SearchPipeline


def normalized_vectors(count: int = 32, dimension: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(count, dimension)).astype("float32")
    faiss.normalize_L2(values)
    return values


class HnswBuildTests(unittest.TestCase):
    def test_build_save_load_and_inner_product_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = normalized_vectors()
            np.save(root / "vectors.npy", vectors)
            info = build_one(
                root / "vectors.npy",
                root / "vectors.index",
                chunk_size=7,
                hnsw_m=8,
                ef_construction=40,
                expected_count=len(vectors),
                expected_dimension=vectors.shape[1],
            )
            index = faiss.read_index(str(root / "vectors.index"))
            index.hnsw.efSearch = 32
            scores, ids = index.search(vectors[:3], 1)
            self.assertEqual(index.ntotal, len(vectors))
            self.assertEqual(index.d, vectors.shape[1])
            self.assertEqual(index.metric_type, faiss.METRIC_INNER_PRODUCT)
            self.assertEqual(index.hnsw.efConstruction, 40)
            self.assertEqual(index.hnsw.nb_neighbors(0), 16)
            self.assertEqual(ids[:, 0].tolist(), [0, 1, 2])
            self.assertTrue(np.allclose(scores[:, 0], 1.0, atol=1e-5))
            self.assertGreater(info["size_bytes"], 0)

    def test_rejects_non_normalized_vectors(self) -> None:
        values = np.ones((4, 3), dtype="float32")
        index = faiss.IndexHNSWFlat(3, 4, faiss.METRIC_INNER_PRODUCT)
        with self.assertRaisesRegex(ValueError, "不是 l2 归一化向量"):
            add_embeddings(index, values, chunk_size=2, desc="bad")

    def test_rejects_invalid_hnsw_parameters(self) -> None:
        args = argparse.Namespace(
            chunk_size=1,
            hnsw_m=0,
            ef_construction=200,
            benchmark_sample_size=1,
            candidate_pool_size=100,
        )
        with self.assertRaisesRegex(ValueError, "hnsw_m"):
            validate_args(args)

    def test_existing_indexes_require_explicit_replace_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = {
                "model_fingerprint": "model",
                "data_manifest_hash": "data",
                "dimension": 8,
                "normalization": "l2",
            }
            (root / "extract_state.json").write_text(
                json.dumps({**contract, "count": 1, "complete": True}), encoding="utf-8"
            )
            (root / "item_meta.json").write_text(
                json.dumps({**contract, "count": 1, "items": [{}]}), encoding="utf-8"
            )
            for name in ("text.index", "image.index", "index_meta.json"):
                (root / name).touch()
            with patch.object(sys, "argv", ["build_faiss_index.py", "--embedding_dir", str(root)]):
                with self.assertRaisesRegex(FileExistsError, "--replace_existing"):
                    main()


class HnswPipelineTests(unittest.TestCase):
    def test_load_hnsw_stays_on_cpu_and_applies_ef_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.faiss"
            index = faiss.IndexHNSWFlat(8, 8, faiss.METRIC_INNER_PRODUCT)
            index.add(normalized_vectors(count=16, dimension=8))
            faiss.write_index(index, str(path))

            pipeline = SearchPipeline.__new__(SearchPipeline)
            pipeline.hnsw_ef_search = 77
            pipeline._hnsw_cpu_notice_printed = False
            pipeline.gpu_resources = []
            loaded = pipeline.load_index(path, use_gpu=True)

            self.assertTrue(hasattr(loaded, "hnsw"))
            self.assertEqual(loaded.hnsw.efSearch, 77)
            self.assertEqual(pipeline.gpu_resources, [])

    def test_metadata_mismatch_is_rejected(self) -> None:
        index = faiss.IndexHNSWFlat(8, 8, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 32
        index.add(normalized_vectors(count=16, dimension=8))

        pipeline = SearchPipeline.__new__(SearchPipeline)
        pipeline.hnsw_ef_search = 32
        pipeline.metadata_mode = "strict"
        pipeline.index_meta = {
            "hnsw": {"m": 16, "ef_construction": 40, "ef_search": 32},
            "text": {"count": 16, "dimension": 8, "normalization": "l2"},
        }
        with self.assertRaisesRegex(MetadataMismatchError, "HNSW M"):
            pipeline._validate_one_index("text", index, 16, 8)

    def test_missing_ef_search_is_rejected(self) -> None:
        with self.assertRaisesRegex(MetadataMismatchError, "ef_search"):
            SearchPipeline.resolve_hnsw_ef_search(None, {})


class FinalFusionBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = [
            {"source": "demo", "image_id": "a"},
            {"source": "demo", "image_id": "b"},
            {"source": "demo", "image_id": "c"},
            {"source": "demo", "image_id": "c"},
            {"source": "demo", "image_id": "d"},
            {"source": "demo", "image_id": "e"},
            {"source": "demo", "image_id": "f"},
            {"source": "demo", "image_id": "g"},
            {"source": "demo", "image_id": "h"},
            {"source": "demo", "image_id": "i"},
            {"source": "demo", "image_id": "j"},
        ]
        for item in self.items:
            item["source"] = "MUGE"

    def test_text_image_and_joint_use_expected_weights(self) -> None:
        routes = {
            "tt": {0: 1.0, 1: 0.2},
            "ti": {0: 0.0, 1: 1.0},
            "it": {0: 0.0, 1: 1.0},
            "ii": {0: 1.0, 1: 0.0},
        }
        row_to_text_id = {str(index): f"demo_{item['image_id']}" for index, item in enumerate(self.items)}
        self.assertEqual(final_ranked_keys(routes, self.items, row_to_text_id, "text", 2)[0], "demo_b")
        self.assertEqual(final_ranked_keys(routes, self.items, row_to_text_id, "image", 2)[0], "demo_a")
        self.assertEqual(final_ranked_keys(routes, self.items, row_to_text_id, "joint", 2)[0], "demo_b")

    def test_final_results_are_deduplicated_before_recall(self) -> None:
        routes = {
            "tt": {2: 1.0, 3: 0.9, 0: 0.8, 1: 0.7, 4: 0.6, 5: 0.5},
            "ti": {},
        }
        row_to_text_id = {str(index): f"demo_{item['image_id']}" for index, item in enumerate(self.items)}
        keys = final_ranked_keys(routes, self.items, row_to_text_id, "text", 5)
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys.count("demo_c"), 1)

    def test_recall_is_computed_at_five_and_ten(self) -> None:
        exact = [("row", str(index)) for index in range(10)]
        approximate = exact[:4] + [("row", "99")] + exact[5:9] + [("row", "98")]
        self.assertEqual(recall_at_k(exact, approximate, 5), 0.8)
        self.assertEqual(recall_at_k(exact, approximate, 10), 0.8)

    def test_fixed_ef_search_is_512(self) -> None:
        self.assertEqual(FIXED_EF_SEARCH, 512)


if __name__ == "__main__":
    unittest.main()
