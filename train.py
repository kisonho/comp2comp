"""Fine-tune a pretrained Stanford nnUNet Keras model."""

import argparse, gc, keras, os, random, torch
from collections.abc import Iterator
from typing import Union, cast

from torch.utils.data import DataLoader, Sampler
from torchmanager.configs import Configs as BaseConfigs
from torchmanager_core import view

from comp2comp.finetune.metrics import SparseMeanDice
from comp2comp.finetune.optim import LayerwiseAdam
from data import ProcessedTCIA, TCIA


ArgumentParser = Union[argparse.ArgumentParser, argparse._ArgumentGroup]


class PatientSampler(Sampler[int]):
    """Shuffle patients and slices without interleaving patient volumes."""

    def __init__(self, dataset: ProcessedTCIA) -> None:
        self.dataset = dataset

    def __iter__(self) -> Iterator[int]:
        sample_ranges = list(self.dataset.sample_ranges)
        random.shuffle(sample_ranges)
        for sample_range in sample_ranges:
            indices = list(sample_range)
            random.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return len(self.dataset)


class ClearCudaCache(keras.callbacks.Callback):
    """Release unused cached CUDA memory after each training epoch."""

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TrainingConfigs(BaseConfigs):
    """Command-line configurations for fine-tuning."""

    dataset_dir: str
    val_dataset_dir: str
    pretrained_model: str
    output_model: str
    batch_size: int
    epoch: int
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
        self.val_dataset_dir = os.path.abspath(
            os.path.expanduser(self.val_dataset_dir)
        )
        self.pretrained_model = os.path.abspath(
            os.path.expanduser(self.pretrained_model)
        )
        self.output_model = os.path.abspath(os.path.expanduser(self.output_model))

        if self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if self.epoch < 1:
            raise ValueError("epoch must be a positive integer")

    @staticmethod
    def get_arguments(parser: ArgumentParser = argparse.ArgumentParser()) -> ArgumentParser:
        """Add training arguments to an argument parser."""
        parser.add_argument("dataset_dir", help="Processed training dataset directory")
        parser.add_argument("pretrained_model", help="Pretrained Keras model path")
        parser.add_argument("output_model", help="Fine-tuned Keras model path")

        dataset_args = parser.add_argument_group("Dataset Arguments")
        dataset_args.add_argument(
            "--val_dataset_dir",
            required=True,
            help="Processed validation dataset directory",
        )
        dataset_args.add_argument(
            "-b",
            "--batch_size",
            type=int,
            default=32,
            help="Training batch size (default: 32)",
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
        view.logger.info(f"Dataset directory: {self.dataset_dir}")
        view.logger.info(f"Validation dataset directory: {self.val_dataset_dir}")
        view.logger.info(f"Pretrained model: {self.pretrained_model}")
        view.logger.info(f"Output model: {self.output_model}")
        view.logger.info(f"Training arguments: batch_size={self.batch_size}, epochs={self.epoch}, lr={self.lr}, unfreeze_decoder={self.unfreeze_decoder}, recompile={self.recompile}")


def train(configs: TrainingConfigs) -> keras.Model:
    """Fine-tune and save the configured Keras model."""
    # normalize directory
    dataset_dir = configs.dataset_dir
    val_dataset_dir = configs.val_dataset_dir
    pretrained_model = configs.pretrained_model
    output_model = configs.output_model
    experiment_dir = configs.experiment
    best_ckpt_path = os.path.join("experiments", experiment_dir, "checkpoints", "best.keras")
    last_ckpt_path = os.path.join("experiments", experiment_dir, "checkpoints", "last.keras")
    tensorboard_dir = os.path.join("experiments", experiment_dir, "data")

    # load dataset
    dataset = ProcessedTCIA(dataset_dir) if configs.preprocessed else TCIA(dataset_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        sampler=PatientSampler(dataset),
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_dataset = ProcessedTCIA(val_dataset_dir) if configs.preprocessed else TCIA(val_dataset_dir)
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
    base_model = cast(keras.Model, keras.models.load_model(pretrained_model))
    view.logger.info("Pre-trained model loaded.")
    num_classes = cast(int, base_model.output_shape[-1])
    base_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseMeanDice(num_classes=num_classes),
        ],
    )
    view.logger.info("Testing pre-trained model...")
    pretrained_results = base_model.evaluate(
        val_data_loader,
        return_dict=True,
    )
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
            if isinstance(layer, keras.layers.Conv2DTranspose)
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
    # segmentation_output = keras.layers.Conv2D(
    #     filters=num_classes,
    #     kernel_size=1,
    #     activation="softmax",
    #     name="segmentation_head",
    # )(base_model.layers[-1].input)
    # model = keras.Model(
    #     inputs=base_model.input,
    #     outputs=segmentation_output,
    #     name=base_model.name,
    # )
    if configs.recompile:
        model.compile(
            optimizer=LayerwiseAdam(
                learning_rate=configs.lr,
                variable_lr_multipliers=variable_lr_multipliers,
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                SparseMeanDice(num_classes=num_classes),
            ],
        )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor="val_dice",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=last_ckpt_path,
            save_weights_only=False,
            save_freq="epoch",
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
            profile_batch=0,
        ),
        ClearCudaCache(),
    ]

    # train model
    view.logger.info("Start training...")
    model.fit(
        data_loader,
        epochs=configs.epoch,
        validation_data=val_data_loader,
        callbacks=callbacks,
    )

    model.save(output_model)
    view.logger.info("Training finished!")
    return model


if __name__ == "__main__":
    """Run the training entrypoint."""
    cfgs = cast(TrainingConfigs, TrainingConfigs.from_arguments())
    train(cfgs)
