# Training Pipeline Specification

## Overview

Training pipeline for the OmegaZero chess engine. Supports two training modes:
1. **Supervised pre-training** on master-level games
2. **Self-play refinement** on MCTS-generated data

## Training Phases

### Phase 1: Supervised Pre-training

Train on master-level games to bootstrap a competent network before self-play.

**Data source:** Lichess database (Rapid/Classical games from 2400+ rated players)
- URL: https://database.lichess.org/
- Format: PGN files
- Time controls: Rapid (10+ min) or Classical (30+ min) only
- Rating filter: Both players 2400+ 
- Target: ~1-10M positions for initial training

**Why these criteria:**
- Rapid/Classical: Players have time to calculate properly, fewer time-pressure mistakes
- 2400+: Master-level play with high-quality moves
- Avoid bullet/blitz: Too many inaccuracies even from strong players

**Targets:**
- Policy: One-hot encoding of the move played
- Value: Stockfish evaluation converted to [-1, 1]

**Expected outcome:** ~1500-1700 ELO playing strength

### Phase 2: Self-Play Refinement

Train on self-generated games to improve beyond imitation.

**Data source:** Self-play games from `SelfPlay` class
- Format: (position, MCTS policy, game outcome) tuples

**Targets:**
- Policy: MCTS visit count distribution (improved over raw network)
- Value: Actual game outcome (+1, -1, 0)

## Data Processing

### PGN Parsing

Extract positions and moves from PGN files:

```python
def parse_pgn(
    pgn_path: str,
    min_rating: int = 2400,
    time_controls: list[str] | None = None,
    min_game_length: int = 20
) -> Iterator[tuple[ChessGame, chess.Move]]:
    """
    Yield (position, move_played) pairs from a PGN file.
    
    Args:
        pgn_path: Path to PGN file
        min_rating: Minimum rating for both players
        time_controls: Allowed time controls (e.g., ["rapid", "classical"])
        min_game_length: Skip games shorter than this many moves
    
    Yields:
        game: ChessGame at position before the move
        move: The move that was played
    """
```

### Stockfish Evaluation

Convert centipawn evaluations to value targets:

```python
def cp_to_value(cp: float, k: float = 400.0) -> float:
    """
    Convert centipawn score to value in [-1, 1].
    
    Args:
        cp: Centipawn score (positive = white advantage)
        k: Scaling factor (lower = more extreme values)
    
    Returns:
        Value in [-1, 1] using tanh scaling
    """
    if cp > 10000:
        return 1.0   # Mate for white
    if cp < -10000:
        return -1.0  # Mate for black
    return np.tanh(cp / k)
```

**k parameter:**
- k=500: Conservative, values closer to 0
- k=400: Standard, good spread
- k=300: Aggressive, more extreme values

**Sign convention:** Value is always from the perspective of the player to move (matches `get_canonical_board()`).

### Training Example Creation

```python
@dataclass
class TrainingExample:
    position: np.ndarray      # (8, 8, 18) canonical board
    policy_target: np.ndarray # (4672,) one-hot or distribution
    value_target: float       # [-1, 1]
```

**From master game:**
```python
game = ChessGame(fen=position_fen)
position = game.get_canonical_board()
policy_target = np.zeros(4672)
policy_target[game.get_move_index(move_played)] = 1.0
value_target = cp_to_value(stockfish_eval, k=400)
# Flip value sign if black to move (canonical perspective)
if game.board.turn == chess.BLACK:
    value_target = -value_target
```

**From self-play:**
```python
position = game.get_canonical_board()
policy_target = mcts_policy  # Already normalized visit counts
value_target = game_outcome  # +1, -1, or 0
```

## Data Loading

### TrainingDataset Class

Handles loading and batching of training data.

```python
class TrainingDataset:
    def __init__(
        self,
        examples: list[TrainingExample] | None = None,
        shuffle: bool = True,
        seed: int | None = None
    )
    
    @classmethod
    def from_pgn(
        cls,
        pgn_path: str,
        stockfish_path: str,
        stockfish_depth: int = 15,
        k: float = 400.0,
        min_rating: int = 2400,
        time_controls: list[str] | None = None,
        min_game_length: int = 20,
        max_positions: int | None = None
    ) -> "TrainingDataset"
    
    @classmethod
    def from_self_play(
        cls,
        games_dir: str
    ) -> "TrainingDataset"
    
    def __len__(self) -> int
    
    def get_batches(
        self,
        batch_size: int = 32
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]
    
    def save(self, filepath: str) -> None
    
    @classmethod
    def load(cls, filepath: str) -> "TrainingDataset"
```

### Memory Considerations

M1 Air has 8GB unified memory. For large datasets:
- Process PGN files in chunks
- Save processed examples to disk (numpy `.npz` format)
- Load batches on demand during training

## Trainer Class

Orchestrates the training loop.

```python
class Trainer:
    def __init__(
        self,
        neural_network: ChessNN,
        learning_rate: float = 0.001,
        checkpoint_dir: str = "checkpoints"
    )
    
    def train(
        self,
        dataset: TrainingDataset,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1,
        callbacks: list | None = None
    ) -> dict[str, list[float]]
    
    def save_checkpoint(self, epoch: int) -> str
    
    def load_checkpoint(self, checkpoint_path: str) -> int
```

### Training Loop

```
for epoch in range(epochs):
    for batch in dataset.get_batches(batch_size):
        positions, policy_targets, value_targets = batch
        loss = neural_network.train(positions, policy_targets, value_targets)
    
    # Validation
    val_loss = evaluate_on_validation_set()
    
    # Checkpoint
    if epoch % checkpoint_frequency == 0:
        save_checkpoint(epoch)
    
    # Logging
    log_metrics(epoch, train_loss, val_loss)
```

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| batch_size | 32 | Constrained by M1 memory |
| learning_rate | 0.001 | Adam optimizer |
| epochs | 10 | Per training run |
| validation_split | 0.1 | 10% for validation |
| stockfish_depth | 15 | Balance speed vs accuracy |
| k (centipawn scaling) | 400.0 | Standard spread |
| checkpoint_frequency | 1 | Every epoch |
| min_rating | 2400 | Master-level games |
| time_controls | ["rapid", "classical"] | Skip bullet/blitz |
| min_game_length | 20 | Skip short games |

## Class Interface Summary

### TrainingDataset

```python
class TrainingDataset:
    def __init__(
        self,
        examples: list[TrainingExample] | None = None,
        shuffle: bool = True,
        seed: int | None = None
    )
    
    @classmethod
    def from_pgn(
        cls,
        pgn_path: str,
        stockfish_path: str,
        stockfish_depth: int = 15,
        k: float = 400.0,
        min_rating: int = 2400,
        time_controls: list[str] | None = None,
        min_game_length: int = 20,
        max_positions: int | None = None
    ) -> "TrainingDataset"
    
    @classmethod
    def from_self_play(
        cls,
        games_dir: str
    ) -> "TrainingDataset"
    
    def __len__(self) -> int
    
    def get_batches(
        self,
        batch_size: int = 32
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]
    
    def save(self, filepath: str) -> None
    
    @classmethod
    def load(cls, filepath: str) -> "TrainingDataset"
```

### Trainer

```python
class Trainer:
    def __init__(
        self,
        neural_network: ChessNN,
        learning_rate: float = 0.001,
        checkpoint_dir: str = "checkpoints"
    )
    
    def train(
        self,
        dataset: TrainingDataset,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1,
        callbacks: list | None = None
    ) -> dict[str, list[float]]
    
    def save_checkpoint(self, epoch: int) -> str
    
    def load_checkpoint(self, checkpoint_path: str) -> int
```

### Utility Functions

```python
def cp_to_value(cp: float, k: float = 400.0) -> float
    """Convert centipawn to value in [-1, 1]."""

def parse_pgn(
    pgn_path: str,
    min_rating: int = 2400,
    time_controls: list[str] | None = None,
    min_game_length: int = 20
) -> Iterator[tuple[ChessGame, chess.Move]]
    """Yield (position, move) pairs from PGN file."""
```

## File Formats

### Processed Dataset (.npz)

```python
# Save
np.savez(
    filepath,
    positions=positions_array,      # (N, 8, 8, 18)
    policy_targets=policy_array,    # (N, 4672)
    value_targets=value_array       # (N,)
)

# Load
data = np.load(filepath)
positions = data['positions']
policy_targets = data['policy_targets']
value_targets = data['value_targets']
```

### Checkpoint Directory Structure

```
checkpoints/
├── epoch_001/
│   ├── model.keras
│   └── metadata.json
├── epoch_002/
│   ├── model.keras
│   └── metadata.json
└── best/
    ├── model.keras
    └── metadata.json
```

**metadata.json:**
```json
{
    "epoch": 2,
    "train_loss": 1.234,
    "val_loss": 1.456,
    "train_policy_accuracy": 0.35,
    "train_value_mae": 0.12,
    "timestamp": "2024-01-15T10:30:00"
}
```

## Implementation Notes

- Stockfish must be installed separately (`brew install stockfish` on macOS)
- Use `chess.engine.SimpleEngine` for Stockfish integration
- Process positions in batches for Stockfish (avoid per-position overhead)
- Shuffle training data each epoch for better convergence
- Monitor GPU memory with `tf.config.experimental.get_memory_info`
- Filter PGN games by time control header (e.g., `[TimeControl "600+0"]`)

## Usage Example

```python
from chess_game import ChessGame
from neural_network import ChessNN
from training import TrainingDataset, Trainer

# Build network
nn = ChessNN()
nn.build_model()

# Load master-level Rapid/Classical games
dataset = TrainingDataset.from_pgn(
    pgn_path="data/lichess_2023.pgn",
    stockfish_path="/opt/homebrew/bin/stockfish",
    stockfish_depth=15,
    min_rating=2400,
    time_controls=["rapid", "classical"],
    min_game_length=20,
    max_positions=100000
)

# Save processed dataset for reuse
dataset.save("data/processed/master_100k.npz")

# Train
trainer = Trainer(neural_network=nn, checkpoint_dir="checkpoints")
history = trainer.train(
    dataset=dataset,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Save final model
nn.save_model("models/supervised_v1.keras")
```

## Dependencies

Requires Stockfish for position evaluation:
```bash
# macOS
brew install stockfish

# Verify
which stockfish  # Should show path
```

Add to `pyproject.toml` if not present:
```toml
[project.optional-dependencies]
training = [
    "chess",  # Already included, has engine support
]
```
