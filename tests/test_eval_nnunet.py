"""Run with: KERAS_BACKEND=torch python -m unittest discover -s tests -v."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import nrrd
import numpy as np
import torch

import eval as evaluation
from data.__main__ import _process_patient


class NativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trainer = self.root / "trainer"
        self.fold = self.trainer / "all"
        self.fold.mkdir(parents=True)
        for path in (self.trainer / "plans.pkl", self.fold / "model_final_checkpoint.model",
                     self.fold / "model_final_checkpoint.model.pkl"):
            path.touch()

    def test_checkpoint_paths(self):
        for path in (self.trainer, self.fold, self.fold / "model_final_checkpoint.model"):
            with self.subTest(path=path):
                self.assertEqual(evaluation.resolve_nnunet_checkpoint(
                    path, "all", "model_final_checkpoint"
                ), (self.trainer, "all", "model_final_checkpoint"))
        numbered = self.trainer / "fold_2"
        self.fold.rename(numbered)
        self.assertEqual(evaluation.resolve_nnunet_checkpoint(
            numbered / "model_final_checkpoint.model", "all", "unused"
        ), (self.trainer, [2], "model_final_checkpoint"))

    def test_missing_checkpoint_metadata(self):
        (self.fold / "model_final_checkpoint.model.pkl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "model.pkl"):
            evaluation.resolve_nnunet_checkpoint(self.trainer, "all", "model_final_checkpoint")

    def test_auto_dispatch_and_raw_data_requirement(self):
        with patch.object(evaluation, "evaluate_nnunet", return_value={"dice": 1}) as native:
            self.assertEqual(evaluation.evaluate(self.root, self.trainer), {"dice": 1})
            native.assert_called_once()
            with self.assertRaisesRegex(ValueError, "raw_dataset_dir"):
                evaluation.evaluate(self.root, self.trainer, preprocessed=True)

    def test_global_metrics_and_absent_classes(self):
        counts = np.zeros((3, 3), dtype=np.int64)
        evaluation.update_confusion(counts, np.array([0, 1]), np.array([0, 2]))
        evaluation.update_confusion(counts, np.array([2, 2]), np.array([2, 0]))
        scores = evaluation.confusion_metrics(counts)
        self.assertEqual(scores["accuracy"], .5)
        self.assertAlmostEqual(scores["dice"], (2/3 + 0 + .5) / 3)
        self.assertEqual(evaluation.confusion_metrics(np.diag([3, 0, 0]))["dice"], 1)
        with self.assertRaises(ValueError):
            evaluation.update_confusion(counts, np.array([3]), np.array([0]))

    def test_coordinate_conversion(self):
        image = np.arange(24).reshape(2, 3, 4)
        converted = evaluation.nrrd_to_nifti(image, {
            "space": "left-posterior-superior",
            "space directions": np.diag([.98, .98, 2.5]),
            "space origin": [-250, 249, -437],
        })
        np.testing.assert_allclose(converted.affine, [
            [-.98, 0, 0, 250], [0, -.98, 0, -249], [0, 0, 2.5, -437], [0, 0, 0, 1]
        ])
        np.testing.assert_array_equal(np.asarray(converted.dataobj), image)

    def make_data(self):
        raw = self.root / "raw"
        header = {
            "space": "left-posterior-superior",
            "space directions": np.diag([.98, .98, 2.5]),
            "space origin": [-250, 249, -437],
        }
        for patient in ("a", "b"):
            folder = raw / patient
            folder.mkdir(parents=True)
            # Include unlabeled source slices to exercise slice_indices.npy.
            truth = np.zeros((4, 5, 7), dtype=np.uint8)
            for z in (0, 2, 3, 5, 6):
                truth[1:3, 1:4, z] = 1
                truth[2, 2:4, z] = 2
            native_labels = np.array([0, 3, 4])[truth]
            hu = native_labels.astype(np.float32) * 100 - 1000
            nrrd.write(str(folder / "CT_IMAGE.nrrd"), hu, header)
            nrrd.write(str(folder / "segmented_tissues_final.nrrd"), truth, header)
        return raw

    @staticmethod
    def fake_predict(**kwargs):
        image = nib.load(kwargs["list_of_lists"][0][0])
        hu = np.asarray(image.dataobj)
        assert hu.min() == -1000  # Inference must receive original, unclipped HU.
        labels = np.rint((hu + 1000) / 100).astype(np.uint8)
        # Simulate native inference exporting a differently oriented volume.
        result = nib.as_closest_canonical(nib.Nifti1Image(labels, image.affine))
        nib.save(result, kwargs["output_filenames"][0])

    def test_keras_logits_and_softmax_still_evaluate(self):
        from keras import layers, models
        from torch.utils.data import TensorDataset

        images = torch.zeros((2, 4, 5, 1))
        truth = torch.ones((2, 4, 5), dtype=torch.int64)
        dataset = TensorDataset(images, truth)
        for activation in ("linear", "softmax"):
            with self.subTest(activation=activation):
                model = models.Sequential([
                    layers.Input((4, 5, 1)),
                    layers.Conv2D(3, 1, activation=activation),
                ])
                model.layers[-1].set_weights([
                    np.zeros((1, 1, 1, 3), dtype=np.float32),
                    np.array([0, 10, 0], dtype=np.float32),
                ])
                path = self.root / f"{activation}.keras"
                model.save(path)
                with patch("data.load", return_value=(None, dataset)):
                    scores = evaluation.evaluate(self.root, path, batch_size=1)
                self.assertEqual(scores["accuracy"], 1)
                self.assertEqual(scores["dice"], 1)
                self.assertIn("loss", scores)

    def test_raw_and_preprocessed_score_same_selected_slices(self):
        raw = self.make_data()
        processed = self.root / "processed"
        for patient in sorted(raw.iterdir()):
            _process_patient(patient, processed, torch.device("cpu"))
        for dataset_dir, preprocessed in ((raw, False), (processed, True)):
            with self.subTest(preprocessed=preprocessed):
                with patch("nnunet.inference.predict.predict_cases", side_effect=self.fake_predict) as predict:
                    results = evaluation.evaluate(
                        dataset_dir, self.fold / "model_final_checkpoint.model",
                        preprocessed=preprocessed, raw_dataset_dir=raw, fold=2,
                    )
                # Fold 2 contains last labeled slice of patient a and first of b.
                self.assertEqual(predict.call_count, 2)
                self.assertEqual(results["accuracy"], 1)
                self.assertEqual(results["dice"], 1)
                self.assertNotIn("loss", results)


if __name__ == "__main__":
    unittest.main()
