# Neural Network Specification

## Overview

AlphaZero-style neural network with residual blocks for chess position evaluation and move prediction.

## Architecture

```
Input: (batch, 8, 8, 18)
    ↓
Initial Conv Block
    Conv2D: 128 filters, 3×3, padding='same', use_bias=False
    BatchNormalization
    ReLU
    ↓
Residual Tower (10 blocks)
    ↓
┌─────────────┴─────────────┐
↓                           ↓
Policy Head                 Value Head
    ↓                           ↓
(batch, 4672)               (batch, 1)
```

## Residual Block Structure

```
input (x)
    │
    ├───────────────────────┐  ← Skip connection
    ↓                       │
Conv2D (3×3, use_bias=False)│
    ↓                       │
BatchNormalization          │
    ↓                       │
ReLU                        │
    ↓                       │
Conv2D (3×3, use_bias=False)│
    ↓                       │
BatchNormalization          │
    ↓                       │
Add [conv_output + x] ←─────┘
    ↓
ReLU
    ↓
output
```

## Policy Head Structure

```
Conv2D: 2 filters, 1×1, use_bias=False
BatchNormalization
ReLU
Flatten
Dense: 4672
Softmax
```

Output: Move probability distribution, shape `(batch, 4672)`, sums to 1.0

## Value Head Structure

```
Conv2D: 1 filter, 1×1, use_bias=False
BatchNormalization
ReLU
Flatten
Dense: 256
ReLU
Dense: 1
Tanh
```

Output: Position evaluation, shape `(batch, 1)`, range [-1, 1]

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| num_residual_blocks | 10 | Depth of residual tower |
| num_filters | 128 | Filters per conv layer |
| learning_rate | 0.001 | Adam optimizer initial LR |

Estimated parameters: ~3.6M (verify in notebook)

## Class Interface

```python
class ChessNN:
    def __init__(
        self,
        num_residual_blocks: int = 10,
        num_filters: int = 128,
        learning_rate: float = 0.001
    )
    
    def build_model(self) -> tf.keras.Model
    
    def predict(
        self,
        board_tensor: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]
    
    def train(
        self,
        positions: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        batch_size: int = 32,
        epochs: int = 1,
        validation_split: float = 0.0
    ) -> tf.keras.callbacks.History
    
    def save_weights(self, filepath: str) -> None
    def load_weights(self, filepath: str) -> None
    
    def save_model(self, filepath: str) -> None
    def load_model(self, filepath: str) -> None
```

## Method Specifications

### `build_model()`

Creates and compiles the Keras model. Sets `self.model`.

Compilation:
- Optimizer: Adam with configured learning_rate
- Loss: `{'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'}`
- Metrics: `{'policy': ['accuracy'], 'value': ['mae']}`

### `predict()`

Forward pass for inference.

| Input Shape | Output |
|-------------|--------|
| `(8, 8, 18)` | policy `(4672,)`, value scalar |
| `(N, 8, 8, 18)` | policy `(N, 4672)`, value `(N, 1)` |

- Uses `training=False` for BatchNorm inference mode
- Converts input to float32
- Raises `ValueError` if model not built or invalid shape

### `train()`

Training wrapper around `model.fit()`.

**Parameters:**
- `positions`: Board tensors, shape `(N, 8, 8, 18)`
- `policy_targets`: One-hot move targets, shape `(N, 4672)`
- `value_targets`: Position evaluations, shape `(N,)` or `(N, 1)`, range [-1, 1]
- `batch_size`: Training batch size (default 32)
- `epochs`: Number of training epochs (default 1)
- `validation_split`: Fraction for validation (default 0.0)

**Returns:** Keras History object with training metrics

**Raises:** `ValueError` if model not built or input shapes invalid

### `save_weights()` / `load_weights()`

Save/load model weights only. Requires model to be built first.

Filepath convention: `model.weights.h5`

### `save_model()` / `load_model()`

Save/load complete model (architecture + weights + optimizer state).

Filepath convention: `model.keras`

`load_model()` rebuilds `self.model` from file. Does not require `build_model()` first.

## Implementation Notes

- All Conv2D layers use `use_bias=False` (BatchNorm makes bias redundant)
- Input dtype: float32
- Policy output sums to 1.0 (softmax)
- Value output in range [-1, 1] (tanh)
- Move masking is external (handled by ChessGame.get_legal_moves_mask())

## Usage Example

```python
nn = ChessNN(num_residual_blocks=10, num_filters=128)
nn.build_model()

# Single prediction
board = game.get_canonical_board()  # (8, 8, 18)
policy, value = nn.predict(board)

# Batch prediction
boards = np.stack([g.get_canonical_board() for g in games])  # (N, 8, 8, 18)
policies, values = nn.predict(boards)

# Training
history = nn.train(positions, policy_targets, value_targets, epochs=10)

# Save/load
nn.save_model("checkpoints/model.keras")
nn.load_model("checkpoints/model.keras")
```
