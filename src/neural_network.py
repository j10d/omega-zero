"""
AlphaZero-style neural network with residual blocks for chess.

This module implements the ChessNN class, which provides a deep neural network
for evaluating chess positions and predicting move probabilities.
"""

import numpy as np
import tensorflow as tf


class ChessNN:
    """
    AlphaZero-style neural network with residual blocks.

    Architecture:
    - Initial convolution block
    - Tower of residual blocks
    - Dual heads (policy + value)

    The network takes board representations and outputs:
    - Policy: move probability distribution (4672 possible moves)
    - Value: position evaluation in range [-1, 1]
    """

    def __init__(
        self,
        num_residual_blocks: int = 10,
        num_filters: int = 128,
        learning_rate: float = 0.001
    ):
        """
        Initialize network configuration.

        Args:
            num_residual_blocks: Number of residual blocks in the tower.
            num_filters: Number of convolutional filters per layer.
            learning_rate: Learning rate for optimizer.
        """
        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters
        self.learning_rate = learning_rate
        self.model = None

    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the Keras model.

        Creates the complete AlphaZero architecture:
        - Input layer (8, 8, 18)
        - Initial convolution block
        - Residual tower
        - Policy head → (4672,)
        - Value head → (1,)

        Returns:
            Compiled Keras model.
        """
        # Input
        inputs = tf.keras.Input(shape=(8, 8, 18), name='board_input')

        # Initial conv block
        x = tf.keras.layers.Conv2D(
            filters=self.num_filters,
            kernel_size=3,
            padding='same',
            use_bias=False,
            name='initial_conv'
        )(inputs)
        x = tf.keras.layers.BatchNormalization(name='initial_bn')(x)
        x = tf.keras.layers.ReLU(name='initial_relu')(x)

        # Residual tower
        for i in range(self.num_residual_blocks):
            x = self._build_residual_block(x, self.num_filters, f'res_{i+1}')

        # Heads
        policy = self._build_policy_head(x)
        value = self._build_value_head(x)

        # Create model
        model = tf.keras.Model(
            inputs=inputs,
            outputs={'policy': policy, 'value': value},
            name='chess_nn'
        )

        # Compile model with optimizer and loss functions
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            },
            metrics={
                'policy': ['accuracy'],
                'value': ['mae']
            }
        )

        self.model = model
        return model

    def predict(
        self,
        board_tensor: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict policy and value for board state(s).

        Handles both single boards and batches:
        - Single: (8, 8, 18) → policy (4672,), value scalar
        - Batch: (N, 8, 8, 18) → policy (N, 4672), value (N, 1)

        Args:
            board_tensor: Board representation(s). Shape (8, 8, 18) or (N, 8, 8, 18).

        Returns:
            policy: Move probabilities, sum=1.0. Shape (4672,) or (N, 4672).
            value: Position evaluation in [-1, 1]. Scalar or shape (N, 1).

        Raises:
            ValueError: If model not built or input has invalid shape.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Validate input shape
        if board_tensor.ndim == 3:
            if board_tensor.shape != (8, 8, 18):
                raise ValueError(
                    f"Invalid board shape: {board_tensor.shape}. "
                    f"Expected (8, 8, 18)."
                )
            board_tensor = np.expand_dims(board_tensor, axis=0)
            single_input = True
        elif board_tensor.ndim == 4:
            if board_tensor.shape[1:] != (8, 8, 18):
                raise ValueError(
                    f"Invalid board shape: {board_tensor.shape}. "
                    f"Expected (batch_size, 8, 8, 18)."
                )
            single_input = False
        else:
            raise ValueError(
                f"Invalid number of dimensions: {board_tensor.ndim}. "
                f"Expected 3 (single board) or 4 (batch)."
            )

        # Ensure float32
        board_tensor = board_tensor.astype(np.float32)

        # Forward pass (training=False for inference)
        outputs = self.model(board_tensor, training=False)

        # Extract outputs
        policy = outputs['policy'].numpy()
        value = outputs['value'].numpy()

        # Squeeze if single input
        if single_input:
            policy = policy[0]
            value = value[0, 0]

        return policy, value

    def train(
        self,
        positions: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        batch_size: int = 32,
        epochs: int = 1,
        validation_split: float = 0.0
    ) -> tf.keras.callbacks.History:
        """
        Train the model on provided data.

        Wrapper around model.fit() with input validation.

        Args:
            positions: Board tensors, shape (N, 8, 8, 18).
            policy_targets: One-hot move targets, shape (N, 4672).
            value_targets: Position evaluations, shape (N,) or (N, 1), range [-1, 1].
            batch_size: Training batch size (default 32).
            epochs: Number of training epochs (default 1).
            validation_split: Fraction for validation (default 0.0).

        Returns:
            Keras History object with training metrics.

        Raises:
            ValueError: If model not built or input shapes invalid.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Validate positions shape
        if positions.ndim != 4 or positions.shape[1:] != (8, 8, 18):
            raise ValueError(
                f"Invalid positions shape: {positions.shape}. "
                f"Expected (N, 8, 8, 18)."
            )

        # Validate policy_targets shape
        if policy_targets.ndim != 2 or policy_targets.shape[1] != 4672:
            raise ValueError(
                f"Invalid policy_targets shape: {policy_targets.shape}. "
                f"Expected (N, 4672)."
            )

        # Validate value_targets shape (accept both (N,) and (N, 1))
        if value_targets.ndim == 1:
            value_targets = value_targets.reshape(-1, 1)
        elif value_targets.ndim != 2 or value_targets.shape[1] != 1:
            raise ValueError(
                f"Invalid value_targets shape: {value_targets.shape}. "
                f"Expected (N,) or (N, 1)."
            )

        # Validate matching sample counts
        n_positions = positions.shape[0]
        n_policy = policy_targets.shape[0]
        n_value = value_targets.shape[0]

        if n_positions != n_policy or n_positions != n_value:
            raise ValueError(
                f"Mismatched sample count: positions={n_positions}, "
                f"policy_targets={n_policy}, value_targets={n_value}."
            )

        # Ensure float32
        positions = positions.astype(np.float32)
        policy_targets = policy_targets.astype(np.float32)
        value_targets = value_targets.astype(np.float32)

        # Train
        history = self.model.fit(
            positions,
            {'policy': policy_targets, 'value': value_targets},
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            verbose=0
        )

        return history

    def save_weights(self, filepath: str) -> None:
        """
        Save model weights to file.

        Args:
            filepath: Path to save weights (e.g., 'model.weights.h5').

        Raises:
            ValueError: If model not built.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.model.save_weights(filepath)

    def load_weights(self, filepath: str) -> None:
        """
        Load model weights from file.

        Args:
            filepath: Path to load weights from.

        Raises:
            ValueError: If model not built.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.model.load_weights(filepath)

    def save_model(self, filepath: str) -> None:
        """
        Save complete model (architecture + weights + optimizer state).

        Uses the Keras .keras format for full serialization.

        Args:
            filepath: Path to save model (e.g., 'model.keras').

        Raises:
            ValueError: If model not built.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load complete model from file.

        Rebuilds self.model from file. Does not require build_model() first.

        Args:
            filepath: Path to load model from.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        self.model = tf.keras.models.load_model(filepath)

    def _build_residual_block(
        self,
        x: tf.Tensor,
        num_filters: int,
        name: str
    ) -> tf.Tensor:
        """
        Build single residual block with skip connection.

        Structure:
            input (x)
                ↓
            ├───────────────────────┐  ← Skip connection
            ↓                       │
            Conv2D (3×3, use_bias=False)
            ↓                       │
            BatchNormalization      │
            ↓                       │
            ReLU                    │
            ↓                       │
            Conv2D (3×3, use_bias=False)
            ↓                       │
            BatchNormalization      │
            ↓                       │
            Add [conv_output + x] ←─┘
            ↓
            ReLU
            ↓
            output

        Args:
            x: Input tensor. Shape (batch, 8, 8, filters).
            num_filters: Number of convolutional filters.
            name: Name prefix for layers.

        Returns:
            Output tensor with same shape as input.
        """
        # Save input for skip connection
        skip = x

        # First conv block
        x = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            padding='same',
            use_bias=False,
            name=f'{name}_conv1'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = tf.keras.layers.ReLU(name=f'{name}_relu1')(x)

        # Second conv block
        x = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=3,
            padding='same',
            use_bias=False,
            name=f'{name}_conv2'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)

        # Skip connection
        x = tf.keras.layers.Add(name=f'{name}_add')([x, skip])

        # Final activation
        x = tf.keras.layers.ReLU(name=f'{name}_relu2')(x)

        return x

    def _build_policy_head(self, x: tf.Tensor) -> tf.Tensor:
        """
        Build policy head.

        Structure:
            Conv2D (1×1, 2 filters, use_bias=False)
            BatchNormalization
            ReLU
            Flatten
            Dense (4672)
            Softmax

        Args:
            x: Input tensor from residual tower. Shape (batch, 8, 8, filters).

        Returns:
            Policy output. Shape (batch, 4672).
        """
        x = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=1,
            use_bias=False,
            name='policy_conv'
        )(x)
        x = tf.keras.layers.BatchNormalization(name='policy_bn')(x)
        x = tf.keras.layers.ReLU(name='policy_relu')(x)
        x = tf.keras.layers.Flatten(name='policy_flatten')(x)
        x = tf.keras.layers.Dense(4672, name='policy_dense')(x)
        policy = tf.keras.layers.Softmax(name='policy_output')(x)

        return policy

    def _build_value_head(self, x: tf.Tensor) -> tf.Tensor:
        """
        Build value head.

        Structure:
            Conv2D (1×1, 1 filter, use_bias=False)
            BatchNormalization
            ReLU
            Flatten
            Dense (256)
            ReLU
            Dense (1)
            Tanh

        Args:
            x: Input tensor from residual tower. Shape (batch, 8, 8, filters).

        Returns:
            Value output. Shape (batch, 1).
        """
        x = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            use_bias=False,
            name='value_conv'
        )(x)
        x = tf.keras.layers.BatchNormalization(name='value_bn')(x)
        x = tf.keras.layers.ReLU(name='value_relu1')(x)
        x = tf.keras.layers.Flatten(name='value_flatten')(x)
        x = tf.keras.layers.Dense(256, name='value_dense1')(x)
        x = tf.keras.layers.ReLU(name='value_relu2')(x)
        x = tf.keras.layers.Dense(1, name='value_dense2')(x)
        value = tf.keras.layers.Activation('tanh', name='value_output')(x)

        return value
