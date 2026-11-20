import keras


@keras.saving.register_keras_serializable(package="comp2comp")
class LayerwiseAdam(keras.optimizers.Adam):
    """Adam optimizer with per-variable learning-rate multipliers."""

    def __init__(
        self,
        variable_lr_multipliers: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.variable_lr_multipliers = variable_lr_multipliers or {}

    def update_step(self, gradient, variable, learning_rate) -> None:
        variable_path = getattr(variable, "path", variable.name)
        multiplier = self.variable_lr_multipliers.get(variable_path, 1.0)
        super().update_step(gradient, variable, learning_rate * multiplier)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "variable_lr_multipliers": self.variable_lr_multipliers,
        }


__all__ = ["LayerwiseAdam"]
