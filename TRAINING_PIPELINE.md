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

## Storage Format

### Compact Storage (Store on Disk)

Positions are stored compactly to minimize disk usage. For 5M positions:
- Pre-computed tensors: ~115 GB (too large)
- Compact format: ~350-450 MB (manageable)

```python
@dataclass
class StoredExample:
    fen: str           # Position as FEN (~60 bytes)
    move_index: int    # Policy target as single index (4 bytes)
    value: float       # Value target in [-1, 1] (4 bytes)
```

**For self-play data**, policy is a distribution, not a single move:

```python
@dataclass
class StoredSelfPlayExample:
    fen: str                    # Position as FEN
    policy: dict[int, float]    # Sparse: {move_index: probability}
    value: float                # Game outcome
```

### Expanded Format (At Training Time)

Convert to tensors when loading batches:

```python
@dataclass
class TrainingExample:
    position: np.ndarray      # (8, 8, 18) canonical board
    policy_target: np.ndarray # (4672,) one-hot or distribution
    value_target: float       # [-1, 1]
```

**Expansion at batch load time:**

```python
def expand_batch(stored_examples: list[StoredExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert compact storage to training tensors."""
    n = len(stored_examples)
    positions = np.zeros((n, 8, 8, 18), dtype=np.float32)
    policies = np.zeros((n, 4672), dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    
    for i, ex in enumerate(stored_examples):
        game = ChessGame(fen=ex.fen)
        positions[i] = game.get_canonical_board()
        policies[i, ex.move_index] = 1.0
        values[i] = ex.value
    
    return positions, policies, values
```

## Data Processing

### PGN Parsing

Extract positions and moves from PGN files:

```python
def parse_pgn(
    pgn_path: str,
    min_rating: int = 2400,
    time_controls: list[str] | None = None,
    min_game_length: int = 20
) -> Iterator[tuple[str, chess.Move]]:
    """
    Yield (fen, move_played) pairs from a PGN file.
    
    Args:
        pgn_path: Path to PGN file
        min_rating: Minimum rating for both players
        time_controls: Allowed time controls (e.g., ["rapid", "classical"])
        min_game_length: Skip games shorter than this many moves
    
    Yields:
        fen: Position as FEN string
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

**From master game:**
```python
fen = board.fen()
game = ChessGame(fen=fen)
move_index = game.get_move_index(move_played)
value = cp_to_value(stockfish_eval, k=400)
# Flip value sign if black to move (canonical perspective)
if game.board.turn == chess.BLACK:
    value = -value

stored = StoredExample(fen=fen, move_index=move_index, value=value)
```

**From self-play:**
```python
fen = game.get_state()
# mcts_policy is sparse dict from MCTS get_policy()
policy = {idx: prob for idx, prob in enumerate(mcts_policy) if prob > 0}
value = game_outcome  # +1, -1, or 0

stored = StoredSelfPlayExample(fen=fen, policy=policy, value=value)
```

## Data Loading

### TrainingDataset Class

Handles loading and batching of training data with on-the-fly expansion.

```python
class TrainingDataset:
    def __init__(
        self,
        examples: list[StoredExample] | None = None,
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

M1 Air has 8GB unified memory. Storage strategy:
- Store compactly on disk (FEN + move_index + value)
- Expand to tensors only at batch load time
- Process PGN files in chunks
- 5M positions ≈ 350-450 MB on disk (very manageable)

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
        positions, policy_targets, value_targets = batch  # Expanded on-the-fly
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
        examples: list[StoredExample] | None = None,
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
) -> Iterator[tuple[str, chess.Move]]
    """Yield (fen, move) pairs from PGN file."""
```

## File Formats

### Processed Dataset (.json or .jsonl)

Compact storage using JSON Lines format (one example per line):

```json
{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "move_index": 847, "value": 0.12}
{"fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "move_index": 1234, "value": 0.08}
```

**Load/Save:**
```python
# Save
with open(filepath, 'w') as f:
    for ex in examples:
        f.write(json.dumps({"fen": ex.fen, "move_index": ex.move_index, "value": ex.value}) + "\n")

# Load
examples = []
with open(filepath, 'r') as f:
    for line in f:
        data = json.loads(line)
        examples.append(StoredExample(**data))
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
- FEN-to-tensor expansion happens at batch load time, not storage time

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

# Save processed dataset for reuse (compact format)
dataset.save("data/processed/master_100k.jsonl")

# Later: load pre-processed dataset
dataset = TrainingDataset.load("data/processed/master_100k.jsonl")

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
