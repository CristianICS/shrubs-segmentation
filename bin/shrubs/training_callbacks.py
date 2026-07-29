import math

import keras
import keras.backend as K


class WarmupCosineScheduler(keras.callbacks.Callback):
    """
    Linear warmup followed by cosine decay.

    Warmup:
        initial_lr = base_lr / warmup_factor
        initial_lr -> base_lr

    Cosine:
        base_lr -> minimum_lr
    """

    def __init__(
        self,
        base_lr=0.0025,
        total_iterations=4000,
        warmup_iterations=2000,
        warmup_factor=10.0,
        minimum_lr=0.0,
        verbose=0,
    ):
        super().__init__()

        if total_iterations <= 0:
            raise ValueError("total_iterations must be positive.")

        if not 0 <= warmup_iterations < total_iterations:
            raise ValueError(
                "warmup_iterations must be between 0 and "
                "total_iterations - 1."
            )

        if warmup_factor <= 0:
            raise ValueError("warmup_factor must be positive.")

        self.base_lr = float(base_lr)
        self.total_iterations = int(total_iterations)
        self.warmup_iterations = int(warmup_iterations)
        self.warmup_factor = float(warmup_factor)
        self.minimum_lr = float(minimum_lr)
        self.verbose = verbose

        self.learning_rates = []

    def calculate_learning_rate(self, iteration):
        """Calculate LR for a zero-based optimizer iteration."""

        iteration = min(iteration, self.total_iterations)

        # Linear warmup
        if iteration < self.warmup_iterations:
            warmup_start_lr = self.base_lr / self.warmup_factor

            warmup_progress = (
                iteration / float(self.warmup_iterations)
                if self.warmup_iterations > 0
                else 1.0
            )

            return warmup_start_lr + warmup_progress * (
                self.base_lr - warmup_start_lr
            )

        # Cosine decay after warmup
        cosine_iterations = (
            self.total_iterations - self.warmup_iterations
        )

        cosine_progress = (
            iteration - self.warmup_iterations
        ) / float(cosine_iterations)

        cosine_progress = min(max(cosine_progress, 0.0), 1.0)

        cosine_value = 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )

        return self.minimum_lr + (
            self.base_lr - self.minimum_lr
        ) * cosine_value

    def _get_optimizer_iteration(self):
        """
        Read the optimizer iteration counter.

        Using optimizer.iterations makes resumed training continue from
        the optimizer's iteration when that state is available.
        """
        return int(K.get_value(self.model.optimizer.iterations))

    def _set_learning_rate(self, learning_rate):
        optimizer = self.model.optimizer

        # Keras 2.3.1 normally exposes optimizer.lr.
        if hasattr(optimizer, "lr"):
            K.set_value(optimizer.lr, learning_rate)
        elif hasattr(optimizer, "learning_rate"):
            K.set_value(optimizer.learning_rate, learning_rate)
        else:
            raise AttributeError(
                "The optimizer has neither 'lr' nor 'learning_rate'."
            )

    def on_train_batch_begin(self, batch, logs=None):
        iteration = self._get_optimizer_iteration()

        learning_rate = self.calculate_learning_rate(iteration)
        self._set_learning_rate(learning_rate)
        self.learning_rates.append(learning_rate)

        if self.verbose and iteration % 100 == 0:
            print(
                "\nIteration {:d}: learning rate = {:.8f}".format(
                    iteration,
                    learning_rate,
                )
            )
