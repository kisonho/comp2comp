import os

from totalsegmentator.libs import download_pretrained_weights, nostdout


try:
    from totalsegmentator.libs import setup_nnunet
except ImportError:
    from totalsegmentator.config import get_weights_dir

    def setup_nnunet():
        weights_dir = get_weights_dir()
        weights_dir.mkdir(exist_ok=True, parents=True)
        os.environ["nnUNet_raw_data_base"] = str(weights_dir)
        os.environ["nnUNet_preprocessed"] = str(weights_dir)
        os.environ["RESULTS_FOLDER"] = str(weights_dir)
