import enum
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Dict, Sequence

import wget
from keras.models import load_model


def _normalize_keras_config_for_legacy_runtime(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "InputLayer":
            config = obj.get("config", {})
            if "batch_shape" in config and "batch_input_shape" not in config:
                config["batch_input_shape"] = config.pop("batch_shape")
            config.pop("optional", None)
        for value in obj.values():
            _normalize_keras_config_for_legacy_runtime(value)
    elif isinstance(obj, list):
        for value in obj:
            _normalize_keras_config_for_legacy_runtime(value)


def _legacy_compatible_keras_archive(filename):
    if not str(filename).endswith(".keras"):
        return filename

    with zipfile.ZipFile(filename, "r") as source:
        config = json.loads(source.read("config.json"))
        before = json.dumps(config, sort_keys=True)
        _normalize_keras_config_for_legacy_runtime(config)
        after = json.dumps(config, sort_keys=True)
        if before == after:
            return filename

        patched = tempfile.NamedTemporaryFile(
            prefix=f"{Path(filename).stem}_", suffix=".keras", delete=False
        )
        patched.close()
        with zipfile.ZipFile(patched.name, "w") as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "config.json":
                    data = json.dumps(config).encode("utf-8")
                target.writestr(item, data)
    return patched.name


def _install_keras_src_functional_alias():
    """Allow Keras 3 archives to load in Keras builds without ``keras.src``."""
    existing = sys.modules.get("keras.src.models.functional")
    if existing is not None and hasattr(existing, "Functional"):
        return

    try:
        if importlib.util.find_spec("keras.src.models.functional") is not None:
            return
    except (ModuleNotFoundError, ValueError):
        pass

    functional_cls = None
    for module_name in (
        "keras.engine.functional",
        "tensorflow.keras.engine.functional",
        "tensorflow.python.keras.engine.functional",
    ):
        try:
            module = importlib.import_module(module_name)
            functional_cls = getattr(module, "Functional", None)
        except Exception:
            continue
        if functional_cls is not None:
            break

    if functional_cls is None:
        from keras.models import Model

        functional_cls = Model

    keras_src = sys.modules.setdefault("keras.src", types.ModuleType("keras.src"))
    keras_src_models = sys.modules.setdefault(
        "keras.src.models", types.ModuleType("keras.src.models")
    )
    keras_src_functional = types.ModuleType("keras.src.models.functional")
    keras_src_functional.__spec__ = importlib.util.spec_from_loader(
        "keras.src.models.functional", loader=None
    )
    keras_src_functional.Functional = functional_cls
    keras_src.models = keras_src_models
    keras_src_models.functional = keras_src_functional
    sys.modules["keras.src.models.functional"] = keras_src_functional


class Models(enum.Enum):
    ABCT_V_0_0_1 = (
        1,
        "abCT_v0.0.1",
        {"muscle": 0, "imat": 1, "vat": 2, "sat": 3},
        False,
        ("soft", "bone", "custom"),
    )

    STANFORD_V_0_0_1 = (
        2,
        "stanford_v0.0.1",
        # ("background", "muscle", "bone", "vat", "sat", "imat"),
        # Category name mapped to channel index
        {"muscle": 1, "vat": 3, "sat": 4, "imat": 5},
        True,
        ("soft", "bone", "custom"),
    )

    STANFORD_V_0_0_2 = (
        3, 
        "stanford_v0.0.2",
        {"muscle": 4, "sat": 1, "vat": 2, "imat": 3},
        True,
        ("soft", "bone", "custom"),
    )
    TCIA = (
        8,
        "tcia",
        {"muscle": 1, "vat": 3, "sat": 4, "imat": 5},
        True,
        ("soft", "bone", "custom"),
    )
    TS_SPINE_FULL = (
        4,
        "ts_spine_full",
        # Category name mapped to channel index
        {
            "L5": 18,
            "L4": 19,
            "L3": 20,
            "L2": 21,
            "L1": 22,
            "T12": 23,
            "T11": 24,
            "T10": 25,
            "T9": 26,
            "T8": 27,
            "T7": 28,
            "T6": 29,
            "T5": 30,
            "T4": 31,
            "T3": 32,
            "T2": 33,
            "T1": 34,
            "C7": 35,
            "C6": 36,
            "C5": 37,
            "C4": 38,
            "C3": 39,
            "C2": 40,
            "C1": 41,
        },
        False,
        (),
    )
    TS_SPINE = (
        5,
        "ts_spine",
        # Category name mapped to channel index
        # {"L5": 18, "L4": 19, "L3": 20, "L2": 21, "L1": 22, "T12": 23},
        {"L5": 27, "L4": 28, "L3": 29, "L2": 30, "L1": 31, "T12": 32},
        False,
        (),
    )
    STANFORD_SPINE_V_0_0_1 = (
        6,
        "stanford_spine_v0.0.1",
        # Category name mapped to channel index
        {"L5": 24, "L4": 23, "L3": 22, "L2": 21, "L1": 20, "T12": 19},
        False,
        (),
    )
    TS_HIP = (
        7,
        "ts_hip",
        # Category name mapped to channel index
        {"femur_left": 88, "femur_right": 89},
        False,
        (),
    )

    def __new__(
        cls,
        value: int,
        model_name: str,
        categories: Dict[str, int],
        use_softmax: bool,
        windows: Sequence[str],
    ):
        obj = object.__new__(cls)
        obj._value_ = value

        obj.model_name = model_name
        obj.categories = categories
        obj.use_softmax = use_softmax
        obj.windows = windows
        return obj

    def load_model(self, model_dir):
        """Load the model from the models directory.

        Args:
            logger (logging.Logger): Logger.

        Returns:
            keras.models.Model: Model.
        """
        try:
            filename = Models.find_model_weights(self.model_name, model_dir)
        except Exception:
            print("Downloading muscle/fat model from hugging face")
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            wget.download(
                f"https://huggingface.co/stanfordmimi/stanford_abct_v0.0.1/resolve/main/{self.model_name}.h5",
                out=os.path.join(model_dir, f"{self.model_name}.h5"),
            )
            filename = Models.find_model_weights(self.model_name, model_dir)
            print("")

        print("Loading muscle/fat model from {}".format(filename))
        _install_keras_src_functional_alias()
        return load_model(_legacy_compatible_keras_archive(filename), compile=False)

    @staticmethod
    def model_from_name(model_name):
        """Get the model enum from the model name.

        Args:
            model_name (str): Model name.

        Returns:
            Models: Model enum.
        """
        for model in Models:
            if model.model_name == model_name:
                return model
        return None

    @staticmethod
    def find_model_weights(file_name, model_dir):
        fallback_filename = None
        exact_filenames = {f"{file_name}.h5", f"{file_name}.keras"}
        for root, _, files in os.walk(model_dir):
            for file in sorted(files):
                if file in exact_filenames:
                    return os.path.join(root, file)
                if file.startswith(file_name):
                    fallback_filename = os.path.join(root, file)
        if fallback_filename is None:
            raise FileNotFoundError(f"No model weights found for {file_name}")
        return fallback_filename
