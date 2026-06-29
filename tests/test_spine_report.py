import sys
import types
import unittest
from pathlib import Path


if "wget" not in sys.modules:
    wget_stub = types.ModuleType("wget")
    wget_stub.download = lambda *args, **kwargs: None
    sys.modules["wget"] = wget_stub

if "nibabel" not in sys.modules:
    nibabel_stub = types.ModuleType("nibabel")
    nibabel_stub.load = lambda *args, **kwargs: None
    nibabel_stub.save = lambda *args, **kwargs: None
    sys.modules["nibabel"] = nibabel_stub

if "totalsegmentator" not in sys.modules:
    totalsegmentator_stub = types.ModuleType("totalsegmentator")
    python_api_stub = types.ModuleType("totalsegmentator.python_api")
    libs_stub = types.ModuleType("totalsegmentator.libs")
    config_stub = types.ModuleType("totalsegmentator.config")

    python_api_stub.totalsegmentator = lambda *args, **kwargs: None
    libs_stub.download_pretrained_weights = lambda *args, **kwargs: None
    libs_stub.nostdout = lambda *args, **kwargs: None
    libs_stub.setup_nnunet = lambda *args, **kwargs: None
    config_stub.get_weights_dir = lambda: "/tmp"

    sys.modules["totalsegmentator"] = totalsegmentator_stub
    sys.modules["totalsegmentator.python_api"] = python_api_stub
    sys.modules["totalsegmentator.libs"] = libs_stub
    sys.modules["totalsegmentator.config"] = config_stub


from comp2comp.spine.spine import SpineMuscleAdiposeTissueReport


class SpineMuscleAdiposeTissueReportTests(unittest.TestCase):
    def test_level_images_include_all_segmented_spine_levels(self):
        report = SpineMuscleAdiposeTissueReport()

        image_files = report._find_level_image_files(
            Path("/tmp/images"),
            levels=["L5", "T11", "C1", "T12", "L1"],
        )

        self.assertEqual(
            [path.name for path in image_files],
            ["C1.png", "T11.png", "T12.png", "L1.png", "L5.png"],
        )


if __name__ == "__main__":
    unittest.main()
