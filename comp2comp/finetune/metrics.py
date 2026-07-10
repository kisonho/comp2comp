import keras


@keras.saving.register_keras_serializable(package="comp2comp")
class SparseMeanDice(keras.metrics.Metric):
    """Mean per-class Dice score for sparse segmentation masks."""

    def __init__(
        self,
        num_classes: int,
        name: str = "dice",
        dtype: str | None = None,
    ) -> None:
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        state_shape = (num_classes,)
        self.intersections = self.add_weight(
            name="intersections", shape=state_shape, initializer="zeros"
        )
        self.true_counts = self.add_weight(
            name="true_counts", shape=state_shape, initializer="zeros"
        )
        self.pred_counts = self.add_weight(
            name="pred_counts", shape=state_shape, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        del sample_weight
        y_true = keras.ops.cast(keras.ops.reshape(y_true, (-1,)), "int32")
        y_pred = keras.ops.cast(
            keras.ops.reshape(keras.ops.argmax(y_pred, axis=-1), (-1,)),
            "int32",
        )
        matching_classes = keras.ops.where(
            keras.ops.equal(y_true, y_pred), y_true, self.num_classes
        )
        self.intersections.assign_add(
            keras.ops.cast(
                keras.ops.bincount(
                    matching_classes, minlength=self.num_classes + 1
                )[: self.num_classes],
                self.dtype,
            )
        )
        self.true_counts.assign_add(
            keras.ops.cast(
                keras.ops.bincount(y_true, minlength=self.num_classes), self.dtype
            )
        )
        self.pred_counts.assign_add(
            keras.ops.cast(
                keras.ops.bincount(y_pred, minlength=self.num_classes), self.dtype
            )
        )

    def result(self):
        denominators = self.true_counts + self.pred_counts
        dice = keras.ops.divide_no_nan(2 * self.intersections, denominators)
        present = keras.ops.cast(denominators > 0, self.dtype)
        return keras.ops.divide_no_nan(
            keras.ops.sum(dice * present), keras.ops.sum(present)
        )

    def get_config(self) -> dict:
        return {**super().get_config(), "num_classes": self.num_classes}


__all__ = ["SparseMeanDice"]