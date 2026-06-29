import numpy as np, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from comp2comp.models.models import Models
from comp2comp.muscle_adipose_tissue import (
    fda_muscle_adipose_tissue,
    muscle_adipose_tissue,
)
from comp2comp.muscle_adipose_tissue.voxtell_utils import (
    VOXTELL_CATEGORIES,
    VOXTELL_MODEL_NAME,
    VOXTELL_PROMPTS,
    align_voxtell_segmentation,
    align_voxtell_segmentation_to_image,
    combine_voxtell_segmentation,
    ensure_voxtell_slice_metadata,
    remap_voxtell_segmentation,
    save_combined_voxtell_segmentation,
    select_voxtell_input_path,
)
from comp2comp.muscle_adipose_tissue.muscle_adipose_tissue_visualization import (
    MuscleAdiposeTissueVisualizer,
)


class VoxTellIntegrationTests(unittest.TestCase):
    def test_model_registry_resolves_voxtell(self):
        model = Models.model_from_name(VOXTELL_MODEL_NAME)
        self.assertIs(model, Models.VOXTELL_V_1_1)
        self.assertEqual(model.categories, VOXTELL_CATEGORIES)

    def test_prompt_order_matches_expected_channels(self):
        self.assertEqual(
            VOXTELL_PROMPTS,
            (
                "oblique muscle",
                "abdominis muscle",
                "psoas muscle",
                "spinalis muscle",
            ),
        )
        self.assertEqual(VOXTELL_CATEGORIES["muscle"], 0)
        self.assertEqual(VOXTELL_CATEGORIES["vat"], 1)
        self.assertEqual(VOXTELL_CATEGORIES["sat"], 2)
        self.assertEqual(VOXTELL_CATEGORIES["imat"], 3)

    def test_remap_voxtell_segmentation_merges_muscle_prompts(self):
        segmentation = np.zeros((4, 3, 5, 7), dtype=np.uint8)
        segmentation[0, 0, 1, 2] = 1
        segmentation[1, 1, 2, 3] = 1
        segmentation[2, 2, 3, 4] = 1
        segmentation[3, 0, 4, 5] = 1

        remapped = remap_voxtell_segmentation(segmentation)

        self.assertEqual(remapped.shape, (4, 3, 5, 7))
        self.assertEqual(remapped[0, 0, 1, 2], 1)
        self.assertEqual(remapped[0, 1, 2, 3], 1)
        self.assertEqual(remapped[0, 2, 3, 4], 1)
        self.assertEqual(remapped[0, 0, 4, 5], 1)
        np.testing.assert_array_equal(remapped[1], np.zeros((3, 5, 7)))
        np.testing.assert_array_equal(remapped[2], np.zeros((3, 5, 7)))
        np.testing.assert_array_equal(remapped[3], np.zeros((3, 5, 7)))

    def test_select_voxtell_input_path_prefers_multilevel_volume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            segmentations_dir = Path(temp_dir) / "segmentations"
            segmentations_dir.mkdir()
            multilevel_path = segmentations_dir / "converted_dcm_multilevel.nii.gz"
            multilevel_path.touch()

            selected_input, selected_output = select_voxtell_input_path(temp_dir)

            self.assertEqual(selected_input, str(multilevel_path))
            self.assertTrue(selected_output.endswith("multilevel_muscle_fat_seg.nii.gz"))

    def test_select_voxtell_input_path_falls_back_to_full_volume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            segmentations_dir = Path(temp_dir) / "segmentations"
            segmentations_dir.mkdir()
            full_volume_path = segmentations_dir / "converted_dcm.nii.gz"
            full_volume_path.touch()

            selected_input, selected_output = select_voxtell_input_path(temp_dir)

            self.assertEqual(selected_input, str(full_volume_path))
            self.assertTrue(selected_output.endswith("muscle_fat_seg.nii.gz"))

    def test_ensure_voxtell_slice_metadata_creates_fallback_names(self):
        pipeline = SimpleNamespace()
        ensure_voxtell_slice_metadata(pipeline, 3)

        self.assertEqual(
            pipeline.dicom_file_names,
            ["slice_0000", "slice_0001", "slice_0002"],
        )
        self.assertEqual(
            pipeline.dicom_file_paths,
            ["slice_0000", "slice_0001", "slice_0002"],
        )

    def test_align_voxtell_segmentation_matches_image_shape(self):
        segmentation = np.zeros((4, 3, 5, 7), dtype=np.uint8)
        segmentation[:, 1, 2, 4] = 1

        aligned = align_voxtell_segmentation(segmentation, (5, 7, 3))

        self.assertEqual(aligned.shape, (4, 5, 7, 3))
        np.testing.assert_array_equal(aligned[:, 2, 4, 1], np.ones(4))

    def test_align_voxtell_segmentation_to_image_preserves_in_plane_axes(self):
        segmentation = np.zeros((4, 3, 5, 7), dtype=np.uint8)
        segmentation[:, 1, 2, 4] = 1

        aligned = align_voxtell_segmentation_to_image(
            segmentation,
            voxtell_shape=(3, 5, 7),
            image_shape=(7, 5, 3),
        )

        self.assertEqual(aligned.shape, (4, 7, 5, 3))
        np.testing.assert_array_equal(aligned[:, 4, 2, 1], np.ones(4))

    def test_align_voxtell_segmentation_to_image_handles_square_slices(self):
        segmentation = np.zeros((4, 12, 512, 512), dtype=np.uint8)
        segmentation[:, 3, 100, 400] = 1

        aligned = align_voxtell_segmentation_to_image(
            segmentation,
            voxtell_shape=(12, 512, 512),
            image_shape=(512, 512, 12),
        )

        np.testing.assert_array_equal(aligned[:, 400, 100, 3], np.ones(4))

    def test_combine_voxtell_segmentation_preserves_prediction_axes(self):
        segmentation = np.zeros((4, 3, 5, 7), dtype=np.uint8)
        segmentation[2, 1, 2, 4] = 1

        combined = combine_voxtell_segmentation(segmentation)

        self.assertEqual(combined.shape, (3, 5, 7))
        self.assertEqual(combined[1, 2, 4], 3)

    def test_save_combined_voxtell_segmentation_uses_reader_writer(self):
        segmentation = np.zeros((4, 3, 5, 7), dtype=np.uint8)
        segmentation[1, 2, 3, 4] = 1
        reader = mock.Mock()
        properties = {"nibabel_stuff": "metadata"}

        save_combined_voxtell_segmentation(
            segmentation, reader, properties, "/tmp/muscle_fat_seg.nii.gz"
        )

        written_seg, written_path, written_properties = reader.write_seg.call_args.args
        self.assertEqual(written_path, "/tmp/muscle_fat_seg.nii.gz")
        self.assertIs(written_properties, properties)
        self.assertEqual(written_seg.shape, (3, 5, 7))
        self.assertEqual(written_seg[2, 3, 4], 2)

    def test_visualizer_has_color_for_full_spine_levels(self):
        visualizer = MuscleAdiposeTissueVisualizer()

        self.assertIn("T11", visualizer._spine_colors)
        self.assertIn("C1", visualizer._spine_colors)

    @mock.patch(
        "comp2comp.muscle_adipose_tissue.muscle_adipose_tissue.ensure_voxtell_slice_metadata"
    )
    @mock.patch(
        "comp2comp.muscle_adipose_tissue.muscle_adipose_tissue.predict_voxtell_volume"
    )
    @mock.patch(
        "comp2comp.muscle_adipose_tissue.muscle_adipose_tissue.ensure_voxtell_checkpoint"
    )
    def test_standard_segmentation_dispatches_to_voxtell(
        self, mock_checkpoint, mock_predict, mock_metadata
    ):
        mock_checkpoint.return_value = "/tmp/models/.voxtell/voxtell_v1.1"
        mock_predict.return_value = (["image"], ["mask"], ["spacing"])

        with tempfile.TemporaryDirectory() as temp_dir:
            segmentations_dir = Path(temp_dir) / "segmentations"
            segmentations_dir.mkdir()
            (segmentations_dir / "converted_dcm_multilevel.nii.gz").touch()

            inference_pipeline = SimpleNamespace(
                output_dir=temp_dir,
                model_dir="/tmp/models",
            )

            segmentation = muscle_adipose_tissue.MuscleAdiposeTissueSegmentation(
                batch_size=1, model_name=VOXTELL_MODEL_NAME
            )
            result = segmentation(inference_pipeline)

        self.assertEqual(result, {"images": ["image"], "preds": ["mask"], "spacings": ["spacing"]})
        mock_checkpoint.assert_called_once_with("/tmp/models")
        mock_predict.assert_called_once()
        self.assertTrue(
            mock_predict.call_args.kwargs["nifti_path"].endswith(
                "converted_dcm_multilevel.nii.gz"
            )
        )
        mock_metadata.assert_called_once_with(inference_pipeline, 1)

    @mock.patch(
        "comp2comp.muscle_adipose_tissue.fda_muscle_adipose_tissue.ensure_voxtell_slice_metadata"
    )
    @mock.patch(
        "comp2comp.muscle_adipose_tissue.fda_muscle_adipose_tissue.predict_voxtell_volume"
    )
    @mock.patch(
        "comp2comp.muscle_adipose_tissue.fda_muscle_adipose_tissue.ensure_voxtell_checkpoint"
    )
    def test_fda_segmentation_dispatches_to_voxtell(
        self, mock_checkpoint, mock_predict, mock_metadata
    ):
        mock_checkpoint.return_value = "/tmp/models/.voxtell/voxtell_v1.1"
        mock_predict.return_value = (["image"], ["mask"], ["spacing"])

        with tempfile.TemporaryDirectory() as temp_dir:
            segmentations_dir = Path(temp_dir) / "segmentations"
            segmentations_dir.mkdir()
            (segmentations_dir / "converted_dcm_multilevel.nii.gz").touch()

            inference_pipeline = SimpleNamespace(
                output_dir=temp_dir,
                model_dir="/tmp/models",
            )

            segmentation = fda_muscle_adipose_tissue.MuscleAdiposeTissueSegmentation(
                batch_size=1, model_name=VOXTELL_MODEL_NAME
            )
            result = segmentation(inference_pipeline)

        self.assertEqual(result, {"images": ["image"], "preds": ["mask"], "spacings": ["spacing"]})
        mock_checkpoint.assert_called_once_with("/tmp/models")
        mock_predict.assert_called_once()
        self.assertTrue(
            mock_predict.call_args.kwargs["nifti_path"].endswith(
                "converted_dcm_multilevel.nii.gz"
            )
        )
        mock_metadata.assert_called_once_with(inference_pipeline, 1)

    @mock.patch.object(
        muscle_adipose_tissue.MuscleAdiposeTissueSegmentation, "forward_pass_2d"
    )
    def test_legacy_segmentation_path_still_uses_keras_backend(self, mock_forward_pass):
        mock_forward_pass.return_value = [
            {"image": "image", "preds": "preds", "spacing": "spacing"}
        ]
        segmentation = muscle_adipose_tissue.MuscleAdiposeTissueSegmentation(
            batch_size=1, model_name="abCT_v0.0.1"
        )
        segmentation.model_type = mock.Mock()
        segmentation.model_type.load_model.return_value = "legacy-model"

        inference_pipeline = SimpleNamespace(
            model_dir="/tmp/models",
            dicom_file_paths=[Path("/tmp/slice_0001.dcm")],
        )

        result = segmentation(inference_pipeline)

        segmentation.model_type.load_model.assert_called_once_with("/tmp/models")
        mock_forward_pass.assert_called_once_with([Path("/tmp/slice_0001.dcm")])
        self.assertEqual(result["images"], ["image"])
        self.assertEqual(result["preds"], ["preds"])
        self.assertEqual(result["spacings"], ["spacing"])


if __name__ == "__main__":
    unittest.main()
