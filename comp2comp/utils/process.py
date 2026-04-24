"""
@author: louisblankemeier
"""

import os
import shutil
import sys
import time
import traceback
from datetime import datetime
import inspect
from pathlib import Path

from comp2comp.io import io_utils


def _configure_torch_checkpoint_loading():
    """Ensure legacy nnUNet checkpoints still load on PyTorch >= 2.6."""
    # Respect explicit user choice when forcing weights-only loading.
    if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "").strip().lower() in {
        "1",
        "true",
    }:
        return

    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # nnUNet/TotalSegmentator checkpoints require full pickle loading.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    # Add known-safe numpy global used by older checkpoints when possible.
    try:
        import numpy as np
        import torch

        scalar = getattr(np.core.multiarray, "scalar", None)
        if scalar is not None and hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([scalar])
    except Exception:
        # Best effort only; env var above is the primary compatibility mechanism.
        pass


def _configure_nnunet_predictor_compat():
    """Bridge TotalSegmentatorV2 to newer nnUNetPredictor signatures.

    Some TotalSegmentatorV2 releases still call nnUNetPredictor with the legacy
    keyword ``perform_everything_on_gpu``. Newer nnunetv2 versions renamed this
    argument to ``perform_everything_on_device``.
    """
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except Exception:
        return

    try:
        signature = inspect.signature(nnUNetPredictor.__init__)
    except Exception:
        return

    params = signature.parameters
    if (
        "perform_everything_on_gpu" in params
        or "perform_everything_on_device" not in params
        or getattr(nnUNetPredictor.__init__, "_comp2comp_compat_wrapped", False)
    ):
        return

    original_init = nnUNetPredictor.__init__

    def compat_init(self, *args, **kwargs):
        if (
            "perform_everything_on_gpu" in kwargs
            and "perform_everything_on_device" not in kwargs
        ):
            kwargs["perform_everything_on_device"] = kwargs.pop(
                "perform_everything_on_gpu"
            )
        else:
            kwargs.pop("perform_everything_on_gpu", None)
        return original_init(self, *args, **kwargs)

    compat_init._comp2comp_compat_wrapped = True
    nnUNetPredictor.__init__ = compat_init


def _configure_nnunet_paths(model_dir):
    """Synchronize nnUNet env vars and imported module globals."""
    model_dir = str(Path(model_dir).resolve())

    os.environ["nnUNet_raw"] = model_dir
    os.environ["nnUNet_preprocessed"] = model_dir
    os.environ["nnUNet_results"] = model_dir

    # Legacy nnUNet/TotalSegmentator variables used by older code paths.
    os.environ.setdefault("nnUNet_raw_data_base", model_dir)
    os.environ.setdefault("RESULTS_FOLDER", model_dir)

    try:
        import nnunetv2.paths as nnunet_paths

        nnunet_paths.nnUNet_raw = model_dir
        nnunet_paths.nnUNet_preprocessed = model_dir
        nnunet_paths.nnUNet_results = model_dir
    except Exception:
        pass

    try:
        import nnunetv2.utilities.dataset_name_id_conversion as dataset_conversion

        dataset_conversion.nnUNet_raw = model_dir
        dataset_conversion.nnUNet_preprocessed = model_dir
        dataset_conversion.nnUNet_results = model_dir
    except Exception:
        pass

    try:
        import nnunetv2.utilities.file_path_utilities as file_path_utilities

        file_path_utilities.nnUNet_results = model_dir
    except Exception:
        pass


def find_common_root(paths):
    paths_with_sep = [path if path.endswith("/") else path + "/" for path in paths]

    # Find common prefix, ensuring it ends with a directory separator
    common_root = os.path.commonprefix(paths_with_sep)
    common_root
    if not common_root.endswith("/"):
        # Find the last separator to correctly identify the common root directory
        common_root = common_root[: common_root.rfind("/") + 1]

    return common_root


def process_2d(args, pipeline_builder):
    _configure_torch_checkpoint_loading()

    output_dir = Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../outputs",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    )
    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True)

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    _configure_nnunet_paths(model_dir)
    _configure_nnunet_predictor_compat()

    pipeline = pipeline_builder(args)

    pipeline(output_dir=output_dir, model_dir=model_dir)


def process_3d(args, pipeline_builder):
    _configure_torch_checkpoint_loading()

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    _configure_nnunet_paths(model_dir)
    _configure_nnunet_predictor_compat()

    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../outputs"
        )

    if not args.overwrite_outputs:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_path, date_time)

    path_and_num = io_utils.get_dicom_or_nifti_paths_and_num(args.input_path)

    # in case input is a .txt file we need to find the common root of the files
    if args.input_path.endswith(".txt"):
        all_paths = [p[0] for p in path_and_num]
        common_root = find_common_root(all_paths)

    for path, num in path_and_num:

        try:
            st = time.time()

            if path.endswith(".nii") or path.endswith(".nii.gz"):
                print("Processing: ", path)
            else:
                print("Processing: ", path, " with ", num, " slices")
                min_slices = 30
                if num < min_slices:
                    print(f"Number of slices is less than {min_slices}, skipping\n")
                    continue

            print("")

            try:
                sys.stdout.flush()
            except Exception:
                pass

            if path.endswith(".nii") or path.endswith(".nii.gz"):
                folder_name = Path(os.path.basename(os.path.normpath(path)))
                # remove .nii or .nii.gz
                folder_name = os.path.normpath(
                    Path(str(folder_name).replace(".gz", "").replace(".nii", ""))
                )
                output_dir = Path(
                    os.path.join(
                        output_path,
                        folder_name,
                    )
                )

            else:
                if args.input_path.endswith(".txt"):
                    output_dir = Path(
                        os.path.join(
                            output_path,
                            os.path.relpath(os.path.normpath(path), common_root),
                        )
                    )
                else:
                    output_dir = Path(
                        os.path.join(
                            output_path,
                            Path(os.path.basename(os.path.normpath(args.input_path))),
                            os.path.relpath(
                                os.path.normpath(path),
                                os.path.normpath(args.input_path),
                            ),
                        )
                    )

            if not os.path.exists(output_dir):
                output_dir.mkdir(parents=True)

            pipeline = pipeline_builder(path, args)

            pipeline(output_dir=output_dir, model_dir=model_dir)

            if not args.save_segmentations:
                # remove the segmentations folder
                segmentations_dir = os.path.join(output_dir, "segmentations")
                if os.path.exists(segmentations_dir):
                    shutil.rmtree(segmentations_dir)

            print(f"Finished processing {path} in {time.time() - st:.1f} seconds\n")
            print("Output was saved to:")
            print(output_dir)

        except Exception:
            print(f"ERROR PROCESSING {path}\n")
            traceback.print_exc()
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            # remove parent folder if empty
            if len(os.listdir(os.path.dirname(output_dir))) == 0:
                shutil.rmtree(os.path.dirname(output_dir))
            continue
