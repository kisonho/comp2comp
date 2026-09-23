"""Fine-tune a pretrained Stanford nnUNet Keras model."""

import argparse, gc, os, random, torch
from collections.abc import Iterator
from keras import activations, callbacks, layers, losses, metrics, models, Model
from typing import Any, Union, cast

from torch.utils.data import DataLoader, Sampler, Subset
from torchmanager.configs import Configs as BaseConfigs
from torchmanager_core import view

import data
from comp2comp.finetune.metrics import SparseMeanDice
from comp2comp.finetune.optim import LayerwiseAdam


ArgumentParser = Union[argparse.ArgumentParser, argparse._ArgumentGroup]


class PatientSampler(Sampler[int]):
    """Shuffle patients and slices without interleaving patient volumes."""

    def __init__(self, dataset: Subset[object]) -> None:
        self.dataset = dataset
        parent_dataset = cast(
            data.ProcessedTCIA | data.TCIA,
            dataset.dataset,
        )
        parent_ranges = parent_dataset.sample_ranges
        parent_to_patient = {
            index: patient_index
            for patient_index, sample_range in enumerate(parent_ranges)
            for index in sample_range
        }
        self.sample_indices: list[list[int]] = [[] for _ in parent_ranges]
        for subset_index, parent_index in enumerate(dataset.indices):
            self.sample_indices[parent_to_patient[parent_index]].append(subset_index)
        self.sample_indices = [indices for indices in self.sample_indices if indices]

    def __iter__(self) -> Iterator[int]:
        sample_indices = list(self.sample_indices)
        random.shuffle(sample_indices)
        for sample in sample_indices:
            indices = list(sample)
            random.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return len(self.dataset)


class ClearCudaCache(callbacks.Callback):
    """Release unused cached CUDA memory after each training epoch."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class KerasLogger(callbacks.Callback):
    """Write Keras epoch metrics through the application logger."""
    params: dict[str, Any]

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        values = logs or {}
        metrics_text = ", ".join(
            f"{name}={float(value):.4f}" for name, value in values.items()
        )
        epoch_count = self.params.get("epochs")
        epoch_text = f"Epoch {epoch + 1}"
        if epoch_count is not None:
            epoch_text += f"/{epoch_count}"
        print()
        view.logger.info(
            f"{epoch_text}: {metrics_text}" if metrics_text else epoch_text
        )


class TrainingConfigs(BaseConfigs):
    """Command-line configurations for fine-tuning."""

    dataset_dir: str
    pretrained_model: str
    output_model: str
    batch_size: int
    epoch: int
    fold: int
    lr: float
    preprocessed: bool
    recompile: bool
    unfreeze_encoder: bool
    unfreeze_decoder: bool
    unfreeze_seg: bool

    def format_arguments(self) -> None:
        """Normalize paths and validate scalar arguments."""
        super().format_arguments()
        self.dataset_dir = os.path.abspath(os.path.expanduser(self.dataset_dir))
        self.pretrained_model = os.path.abspath(
            os.path.expanduser(self.pretrained_model)
        )
        self.output_model = os.path.abspath(os.path.expanduser(self.output_model))

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be a positive integer, got {self.batch_size}.")
        if self.epoch < 1:
            raise ValueError(f"Epoch must be a positive integer, got {self.epoch}.")
        if self.fold > 9 or self.fold < 0:
            raise ValueError(f"Cross validation fold index must be in range [0, 9], got {self.fold}.")

    @staticmethod
    def get_arguments(parser: ArgumentParser = argparse.ArgumentParser()) -> ArgumentParser:
        """Add training arguments to an argument parser."""
        parser.add_argument("dataset_dir", help="Processed training dataset directory")
        parser.add_argument("pretrained_model", help="Pretrained Keras model path")
        parser.add_argument("output_model", help="Fine-tuned Keras model path")

        dataset_args = parser.add_argument_group("Dataset Arguments")
        dataset_args.add_argument(
            "-b",
            "--batch_size",
            type=int,
            default=32,
            help="Training batch size (default: 32)",
        )
        dataset_args.add_argument(
            "-f",
            "--fold",
            type=int,
            default=0,
            help="The cross validation fold index (default: 0)",
        )
        dataset_args.add_argument(
            "--preprocessed",
            action="store_true",
            help="Load the dataset as preprocessed dataset.",
        )

        training_args = parser.add_argument_group("Training Arguments")
        training_args.add_argument(
            "-e",
            "--epoch",
            type=int,
            default=100,
            help="Number of training epochs (default: 100)",
        )
        training_args.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="Training learning rate (default: 1e-3)",
        )
        training_args.add_argument(
            "--unfreeze_encoder",
            action="store_true",
            help="Unfreeze the nnUNet encoder during fine-tuning",
        )
        training_args.add_argument(
            "--unfreeze_decoder",
            action="store_true",
            help="Unfreeze the nnUNet decoder during fine-tuning",
        )
        training_args.add_argument(
            "--unfreeze_seg",
            action="store_true",
            help="Unfreeze the nnUNet segmentation layer during fine-tuning",
        )
        training_args.add_argument(
            "--recompile",
            action="store_true",
            help="Recompile the model after loading.",
        )
        BaseConfigs.get_arguments(parser)
        return parser

    def show_settings(self) -> None:
        """Print the resolved training settings."""
        view.logger.info(f"Dataset: dataset_dir={self.dataset_dir}, fold={self.fold}, preprocessed={self.preprocessed}")
        view.logger.info(f"Pretrained model: {self.pretrained_model}")
        view.logger.info(f"Output model: {self.output_model}")
        view.logger.info(f"Training arguments: batch_size={self.batch_size}, epochs={self.epoch}, lr={self.lr}, unfreeze_seg={self.unfreeze_seg}, unfreeze_decoder={self.unfreeze_decoder}, unfreeze_encoder={self.unfreeze_encoder}, recompile={self.recompile}")


def train(configs: TrainingConfigs) -> Model:
    """Fine-tune and save the configured Keras model."""
    # normalize directory
    dataset_dir = configs.dataset_dir
    pretrained_model = configs.pretrained_model
    output_model = configs.output_model
    experiment_dir = configs.experiment
    best_ckpt_path = os.path.join("experiments", experiment_dir, "checkpoints", "best.keras")
    last_ckpt_path = os.path.join("experiments", experiment_dir, "checkpoints", "last.keras")
    tensorboard_dir = os.path.join("experiments", experiment_dir, "data")

    # load dataset
    dataset, val_dataset = data.load(dataset_dir, preprocessed=configs.preprocessed, fold_idx=configs.fold)
    data_loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        sampler=PatientSampler(dataset),
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # load model
    base_model = cast(Model, models.load_model(pretrained_model))
    view.logger.info("Pre-trained model loaded.")
    num_classes = cast(int, base_model.output_shape[-1])
    output_layer = base_model.layers[-1]
    if not isinstance(output_layer, layers.Conv2D):
        raise TypeError(
            "Expected the segmentation output layer to be Conv2D, got "
            f"{type(output_layer).__name__}."
        )

    # With the Keras Torch backend, SoftmaxBackward can return a strided
    # gradient that Conv2D's CUDA backward kernel rejects with
    # ``RuntimeError: grad_input must be contiguous``. Computing sparse
    # cross-entropy directly from logits avoids that backend-specific layout
    # issue and is also more numerically stable. This assignment is harmless
    # when resuming one of our logits-based training checkpoints.
    output_layer.activation = activations.linear
    training_loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    base_model.compile(
        loss=training_loss,
        metrics=[
            metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseMeanDice(num_classes=num_classes),
        ],
    )
    view.logger.info("Testing pre-trained model...")
    pretrained_results = cast(dict[str, float], base_model.evaluate(
        val_data_loader,
        return_dict=True,
        verbose="0",
    ))
    view.logger.info(
        "Pre-trained test results: "
        + ", ".join(
            f"{name}={float(value):.4f}"
            for name, value in pretrained_results.items()
        )
    )

    # unfreeze layers
    decoder_start = next(
        (
            index
            for index, layer in enumerate(base_model.layers)
            if isinstance(layer, layers.Conv2DTranspose)
        ),
        len(base_model.layers) - 1,
    )
    encoder_range = range(0, decoder_start)
    decoder_range = range(decoder_start, len(base_model.layers) - 1)
    for i in encoder_range:
        base_model.layers[i].trainable = configs.unfreeze_encoder
    for i in decoder_range:
        base_model.layers[i].trainable = configs.unfreeze_decoder
    base_model.layers[-1].trainable = configs.unfreeze_seg

    # compile model
    model = base_model
    variable_lr_multipliers = {}
    for index, layer in enumerate(model.layers):
        if index < decoder_start:
            multiplier = 0.01
        elif index < len(model.layers) - 1:
            multiplier = 0.1
        else:
            multiplier = 1.0
        for variable in layer.weights:
            variable_path = getattr(variable, "path", variable.name)
            variable_lr_multipliers[variable_path] = multiplier
    if configs.recompile:
        model.compile(
            optimizer=LayerwiseAdam(
                learning_rate=configs.lr,
                variable_lr_multipliers=variable_lr_multipliers,
            ),
            loss=training_loss,
            metrics=[
                metrics.SparseCategoricalAccuracy(name="accuracy"),
                SparseMeanDice(num_classes=num_classes),
            ],
        )
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor="val_dice",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        ),
        callbacks.ModelCheckpoint(
            filepath=last_ckpt_path,
            save_weights_only=False,
            save_freq="epoch",
            verbose=0,
        ),
        callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
            profile_batch=0,
        ),
        KerasLogger(),
        ClearCudaCache(),
    ]

    # train model
    view.logger.info("Start training...")
    model.fit(
        data_loader,
        epochs=configs.epoch,
        validation_data=val_data_loader,
        callbacks=callbacks_list,
    )

    model = cast(Model, models.load_model(best_ckpt_path))
    best_results = cast(dict[str, float], model.evaluate(
        val_data_loader,
        return_dict=True,
    ))
    view.logger.info(
        "Best checkpoint validation results: "
        + ", ".join(
            f"{name}={float(value):.4f}"
            for name, value in best_results.items()
        )
    )

    # Keep the exported inference model compatible with the pretrained model:
    # training checkpoints emit logits, while the requested output emits probabilities.
    output_layer = model.layers[-1]
    if not isinstance(output_layer, layers.Conv2D):
        raise TypeError(
            "Expected the segmentation output layer to be Conv2D, got "
            f"{type(output_layer).__name__}."
        )
    output_layer.activation = activations.softmax
    model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[
            metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseMeanDice(num_classes=num_classes),
        ],
    )
    model.save(output_model)
    view.logger.info("Training finished!")
    return model


if __name__ == "__main__":
    """Run the training entrypoint."""
    cfgs = cast(TrainingConfigs, TrainingConfigs.from_arguments())
    train(cfgs)
