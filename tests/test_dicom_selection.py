import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock


if "dosma" not in sys.modules:
    dosma_stub = types.ModuleType("dosma")
    dosma_stub.DicomReader = mock.Mock
    dosma_stub.NiftiWriter = mock.Mock
    dosma_stub.MedicalVolume = object
    sys.modules["dosma"] = dosma_stub

if "nibabel" not in sys.modules:
    nibabel_stub = types.ModuleType("nibabel")
    nibabel_stub.load = mock.Mock()
    sys.modules["nibabel"] = nibabel_stub

if "pydicom" not in sys.modules:
    pydicom_stub = types.ModuleType("pydicom")
    pydicom_filereader_stub = types.ModuleType("pydicom.filereader")
    pydicom_filereader_stub.dcmread = mock.Mock()
    pydicom_stub.filereader = pydicom_filereader_stub
    pydicom_stub.dcmread = mock.Mock()
    sys.modules["pydicom"] = pydicom_stub
    sys.modules["pydicom.filereader"] = pydicom_filereader_stub

if "SimpleITK" not in sys.modules:
    simple_itk_stub = types.ModuleType("SimpleITK")
    simple_itk_stub.ImageSeriesReader = mock.Mock
    simple_itk_stub.WriteImage = mock.Mock()
    sys.modules["SimpleITK"] = simple_itk_stub


from comp2comp.io import io


class DicomSelectionTests(unittest.TestCase):
    def test_select_valid_dicom_files_skips_invalid_initial_file(self):
        def fake_series_selector(dicom_name, pipeline_name=None):
            if dicom_name == "localizer.dcm":
                raise ValueError("Not primary image type")
            return SimpleNamespace(path=dicom_name)

        with mock.patch.object(
            io, "series_selector", side_effect=fake_series_selector
        ):
            dicom_names, ds = io.select_valid_dicom_files(
                ["localizer.dcm", "slice_001.dcm", "slice_002.dcm"]
            )

        self.assertEqual(dicom_names, ["slice_001.dcm", "slice_002.dcm"])
        self.assertEqual(ds.path, "slice_001.dcm")

    def test_select_valid_dicom_files_reports_first_rejection(self):
        with mock.patch.object(
            io,
            "series_selector",
            side_effect=ValueError("Not primary image type"),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "No valid DICOM files found for conversion.*Not primary image type",
            ):
                io.select_valid_dicom_files(["localizer.dcm"])

    def test_series_selector_accepts_float_axial_orientation(self):
        ds = SimpleNamespace(
            ImageType=["ORIGINAL", "PRIMARY", "AXIAL"],
            ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

        with mock.patch.object(io.pydicom.filereader, "dcmread", return_value=ds):
            selected_ds = io.series_selector("slice_001.dcm")

        self.assertIs(selected_ds, ds)


if __name__ == "__main__":
    unittest.main()
