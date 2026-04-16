import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys
import types


if "wget" not in sys.modules:
    wget_stub = types.ModuleType("wget")
    wget_stub.download = lambda *args, **kwargs: None
    sys.modules["wget"] = wget_stub

if "keras" not in sys.modules:
    keras_stub = types.ModuleType("keras")
    keras_backend_stub = types.ModuleType("keras.backend")
    keras_backend_stub.clear_session = lambda: None
    keras_utils_stub = types.ModuleType("keras.utils")
    keras_data_utils_stub = types.ModuleType("keras.utils.data_utils")
    keras_models_stub = types.ModuleType("keras.models")

    class _Sequence:
        pass

    class _OrderedEnqueuer:
        def __init__(self, *args, **kwargs):
            self._generator = iter(())

        def start(self, *args, **kwargs):
            return None

        def get(self):
            return self._generator

    keras_utils_stub.Sequence = _Sequence
    keras_data_utils_stub.OrderedEnqueuer = _OrderedEnqueuer
    keras_models_stub.load_model = lambda *args, **kwargs: mock.Mock()

    keras_stub.backend = keras_backend_stub
    keras_stub.utils = keras_utils_stub

    sys.modules["keras"] = keras_stub
    sys.modules["keras.backend"] = keras_backend_stub
    sys.modules["keras.utils"] = keras_utils_stub
    sys.modules["keras.utils.data_utils"] = keras_data_utils_stub
    sys.modules["keras.models"] = keras_models_stub

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__version__ = "4.0.0"
    cv2_stub.ocl = SimpleNamespace(setUseOpenCL=lambda *args, **kwargs: None)
    cv2_stub.connectedComponentsWithStats = lambda *args, **kwargs: (
        1,
        None,
        [[0, 0, 0, 0, 0]],
        None,
    )
    sys.modules["cv2"] = cv2_stub

if "h5py" not in sys.modules:
    h5py_stub = types.ModuleType("h5py")

    class _File:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def create_dataset(self, *args, **kwargs):
            return None

    h5py_stub.File = _File
    sys.modules["h5py"] = h5py_stub

if "nibabel" not in sys.modules:
    nibabel_stub = types.ModuleType("nibabel")
    nibabel_stub.load = lambda *args, **kwargs: mock.Mock()
    nibabel_stub.save = lambda *args, **kwargs: None
    nibabel_stub.as_closest_canonical = lambda image: image
    nibabel_stub.Nifti1Image = mock.Mock
    sys.modules["nibabel"] = nibabel_stub

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = mock.Mock
    pandas_stub.read_csv = mock.Mock
    sys.modules["pandas"] = pandas_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, **kwargs: iterable if iterable is not None else []
    tqdm_auto_stub = types.ModuleType("tqdm.auto")
    tqdm_auto_stub.tqdm = tqdm_stub.tqdm
    tqdm_contrib_stub = types.ModuleType("tqdm.contrib")
    tqdm_contrib_concurrent_stub = types.ModuleType("tqdm.contrib.concurrent")
    tqdm_contrib_concurrent_stub.thread_map = lambda fn, items, **kwargs: [
        fn(item) for item in items
    ]
    sys.modules["tqdm"] = tqdm_stub
    sys.modules["tqdm.auto"] = tqdm_auto_stub
    sys.modules["tqdm.contrib"] = tqdm_contrib_stub
    sys.modules["tqdm.contrib.concurrent"] = tqdm_contrib_concurrent_stub

if "huggingface_hub" not in sys.modules:
    huggingface_hub_stub = types.ModuleType("huggingface_hub")
    huggingface_hub_stub.snapshot_download = lambda *args, **kwargs: "/tmp/voxtell"
    sys.modules["huggingface_hub"] = huggingface_hub_stub

from comp2comp.models.models import Models
from comp2comp.muscle_adipose_tissue import (
    fda_muscle_adipose_tissue,
    muscle_adipose_tissue,
)
from comp2comp.muscle_adipose_tissue.voxtell_utils import (
    VOXTELL_CATEGORIES,
    VOXTELL_MODEL_NAME,
    VOXTELL_PROMPTS,
    ensure_voxtell_slice_metadata,
    select_voxtell_input_path,
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
                "skeletal muscle",
                "visceral adipose tissue",
                "subcutaneous adipose tissue",
                "intramuscular adipose tissue",
            ),
        )
        self.assertEqual(VOXTELL_CATEGORIES["muscle"], 0)
        self.assertEqual(VOXTELL_CATEGORIES["vat"], 1)
        self.assertEqual(VOXTELL_CATEGORIES["sat"], 2)
        self.assertEqual(VOXTELL_CATEGORIES["imat"], 3)

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
