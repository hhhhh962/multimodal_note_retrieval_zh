"""Document schema v2 and pair-artifact migration tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.migrate_pair_artifacts import migrate
from src.document_schema import (
    build_documents_from_pairs,
    build_retrieval_meta,
    normalize_documents,
    validate_retrieval_meta,
)
from src.retrieval_storage import load_external_metadata


def normalized(values: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype="float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


class DocumentSchemaTests(unittest.TestCase):
    def test_groups_all_sources_and_tracks_shared_images(self) -> None:
        pairs = [
            {"source": "MUGE", "split": "train", "text_id": "t1", "text": "笔记一", "image_id": "i1", "image_path": "a.jpg"},
            {"source": "MUGE", "split": "train", "text_id": "t1", "text": "笔记一", "image_id": "i2", "image_path": "shared.jpg"},
            {"source": "Flickr30k-CN", "split": "train", "text_id": "t2", "text": "笔记二", "image_id": "i2", "image_path": "shared.jpg"},
        ]
        documents, images, pair_docs, pair_images = build_documents_from_pairs(pairs)
        self.assertEqual([document["doc_id"] for document in documents], ["MUGE:train:t1", "Flickr30k-CN:train:t2"])
        self.assertEqual([image["image_path"] for image in documents[0]["images"]], ["a.jpg", "shared.jpg"])
        self.assertEqual(len(images), 2)
        self.assertEqual(images[1]["doc_row_ids"], [0, 1])
        self.assertEqual(pair_docs, [0, 0, 1])
        self.assertEqual(pair_images, [0, 1, 1])

    def test_rejects_conflicting_text_for_one_document(self) -> None:
        pairs = [
            {"source": "MUGE", "split": "train", "text_id": "t1", "text": "甲", "image_path": "a.jpg"},
            {"source": "MUGE", "split": "train", "text_id": "t1", "text": "乙", "image_path": "b.jpg"},
        ]
        with self.assertRaisesRegex(ValueError, "冲突文本"):
            build_documents_from_pairs(pairs)

    def test_normalized_documents_reject_duplicate_ids(self) -> None:
        raw = [
            {"doc_id": "same", "text": "甲", "images": [{"image_path": "a.jpg"}]},
            {"doc_id": "same", "text": "乙", "images": [{"image_path": "b.jpg"}]},
        ]
        with self.assertRaisesRegex(ValueError, "重复 doc_id"):
            normalize_documents(raw)

    def test_old_metadata_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "migrate_pair_artifacts.py"):
            validate_retrieval_meta({"items": []})

    def test_missing_text_id_falls_back_to_zero_row_without_unknown_collision(self) -> None:
        pairs = [
            {"source": "custom", "split": "train", "text": "零", "image_path": "0.jpg"},
            {"source": "custom", "split": "train", "text": "一", "image_path": "1.jpg"},
        ]
        documents, _, _, _ = build_documents_from_pairs(pairs)
        self.assertEqual([row["doc_id"] for row in documents], ["custom:train:0", "custom:train:1"])


class PairMigrationTests(unittest.TestCase):
    def _write_source(self, root: Path, *, mismatch: bool = False) -> Path:
        source = root / "source"
        source.mkdir()
        items = [
            {"item_id": "p0", "source": "MUGE", "split": "train", "text_id": "t1", "text": "笔记一", "image_id": "i1", "image_path": "a.jpg"},
            {"item_id": "p1", "source": "MUGE", "split": "train", "text_id": "t1", "text": "笔记一", "image_id": "i2", "image_path": "shared.jpg"},
            {"item_id": "p2", "source": "Flickr30k-CN", "split": "train", "text_id": "t2", "text": "笔记二", "image_id": "i2", "image_path": "shared.jpg"},
            {"item_id": "p3", "source": "COCO-CN", "split": "valid", "text_id": "t3", "text": "笔记三", "image_id": "i3", "image_path": "c.jpg"},
        ]
        compatibility = {
            "model_fingerprint": "model",
            "data_manifest_hash": "pairs",
            "dimension": 3,
            "normalization": "l2",
        }
        (source / "item_meta.json").write_text(
            json.dumps({**compatibility, "count": len(items), "items": items}), encoding="utf-8"
        )
        (source / "extract_state.json").write_text(
            json.dumps({**compatibility, "count": len(items), "complete": True}), encoding="utf-8"
        )
        text = normalized([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if mismatch:
            text[1] = normalized([[1, 1, 0]])[0]
        image = normalized([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
        np.save(source / "text_embeddings.npy", text)
        np.save(source / "image_embeddings.npy", image)
        return source

    def test_migrates_to_different_document_and_image_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_source(root)
            output = root / "v2"
            report = migrate(source, output)
            self.assertEqual(report["source_pair_count"], 4)
            self.assertEqual(report["document_count"], 3)
            self.assertEqual(report["image_count"], 3)
            self.assertEqual(report["relation_count"], 4)
            meta = json.loads((output / "retrieval_meta.json").read_text(encoding="utf-8"))
            validate_retrieval_meta(meta, output)
            self.assertEqual(meta["schema_version"], 2)
            self.assertEqual(meta["metadata_storage"], "jsonl_offsets_csr_v1")
            documents, images, doc_images = load_external_metadata(meta, output)
            self.assertEqual(documents[0]["doc_id"], "MUGE:train:t1")
            self.assertEqual(len(documents[0]["images"]), 2)
            self.assertEqual(images[1]["doc_row_ids"], [0, 1])
            self.assertEqual(doc_images[0].tolist(), [0, 1])
            documents.close()
            images.close()
            self.assertEqual(np.load(output / "text_embeddings.npy").shape, (3, 3))
            self.assertEqual(np.load(output / "image_embeddings.npy").shape, (3, 3))

    def test_rejects_inconsistent_duplicate_vectors_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_source(root, mismatch=True)
            output = root / "v2"
            with self.assertRaisesRegex(ValueError, "重复文本向量不一致"):
                migrate(source, output)
            self.assertFalse(output.exists())
            self.assertFalse((root / ".v2.tmp").exists())

    def test_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_source(root)
            output = root / "v2"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                migrate(source, output)

    def test_explicit_tolerance_records_nonidentical_duplicate_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_source(root)
            text = np.load(source / "text_embeddings.npy")
            text[1] = normalized([[1.0, 0.0001, 0.0]])[0]
            np.save(source / "text_embeddings.npy", text)
            report = migrate(source, root / "v2", atol=5e-4)
            self.assertEqual(report["nonidentical_text_rows_within_atol"], 1)
            self.assertGreater(report["max_duplicate_text_abs_diff"], 0.0)
            self.assertEqual(report["nonidentical_image_rows_within_atol"], 0)


if __name__ == "__main__":
    unittest.main()
