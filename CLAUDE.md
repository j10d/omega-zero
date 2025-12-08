# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# OmegaZero Chess Engine - Project Documentation

## Project Overview

Educational implementation of an AlphaZero-style chess engine.

**Goals:**
- Build working chess engine using self-play reinforcement learning
- Train locally on MacBook Air M1 GPU
- Reach 1500+ ELO playing strength

## Technical Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow with tensorflow-metal (M1 GPU support)
- **Chess Rules:** python-chess library (handles legal move generation)
- **Testing:** pytest
- **Dependencies:** numpy, chess, tensorflow, tensorflow-metal

## Hardware Constraints

- **Platform:** MacBook Air M1
- **GPU:** Apple Silicon (via tensorflow-metal)
- **Implications:** Smaller networks, fewer MCTS simulations, longer training times

## Project Structure

```
omega-zero/
├── CLAUDE.md                    # This file
├── IMPLEMENTATION_AGENT.md      # Implementation Agent instructions
├── REVIEW_AGENT.md              # Review Agent instructions
├── TEST_GUIDELINES.md           # Test naming and organization conventions
├── EXPERIMENTS.md               # Research experiments (separate track)
├── specs/
│   ├── BOARD_REPRESENTATION.md  # Board tensor specification
│   ├── NEURAL_NETWORK.md        # Neural network specification
│   ├── MCTS.md                  # MCTS specification
│   ├── DATA_PREPARATION.md      # Data preparation specification
│   └── TRAINING_PIPELINE.md     # Training pipeline specification
├── src/
│   ├── chess_game.py           # Game environment & rules engine
│   ├── neural_network.py       # Policy + value network
│   ├── mcts.py                 # Monte Carlo Tree Search
│   ├── data_preparation.py     # Data download and processing
│   ├── training.py             # Training pipeline
│   ├── self_play.py            # Self-play game generation
│   └── evaluation.py           # Model evaluation & arena
├── tests/
│   ├── test_chess_game.py
│   ├── test_neural_network.py
│   ├── test_mcts.py
│   ├── test_data_preparation.py
│   └── test_integration.py
└── data/
    ├── raw/                    # Downloaded PGN files
    └── processed/              # Processed training data
```

## Development Commands

```bash
# Setup
python3.10 -m venv venv
source venv/bin/activate
pip install -e ".[dev,metal]"

# Testing
pytest                          # Run all tests
pytest tests/test_chess_game.py # Run specific test file
pytest -v                       # Verbose output
pytest --cov=src                # With coverage
```

## Current Status

**Component Status:**
1. ✅ Game Environment (chess_game.py) - Complete
2. ✅ Neural Network (neural_network.py) - Complete
3. ✅ MCTS (mcts.py) - Complete
4. ✅ Data Preparation (data_preparation.py) - Complete
5. ⏳ Training Pipeline (training.py)
6. ⏳ Self-Play Engine (self_play.py)
7. ⏳ Evaluation System (evaluation.py)

## Milestones

**Milestone 1: Supervised Pre-training**
- Components needed: ChessGame ✅, ChessNN ✅, Data Preparation ✅, Training Pipeline ⏳
- Checkpoint: Train on balanced positions from Lichess database
- Expected result: ~1500-1700 ELO baseline model

**Milestone 2: Minimum Viable Self-Play Loop**
- Components needed: ChessGame ✅, ChessNN ✅, MCTS ✅, Training Pipeline ⏳, Self-Play ⏳
- Checkpoint: Once self-play can generate games and train the network
- Action: Run leaf parallelization benchmark
  - Measure: simulations/second, games/hour with simple MCTS
  - Estimate theoretical speedup from batched leaf evaluation
  - Decide if parallelization complexity is justified for M1 hardware
  - Document findings in EXPERIMENTS.md

## Development Approach

### Test-Driven Development (TDD)
1. Write tests first
2. Implement to pass tests
3. Refactor while keeping tests green
4. Iterate component by component

### Three-Agent Workflow
- **Architecture Agent:** Designs components, provides specifications
- **Implementation Agent:** Writes tests first, then implements (see IMPLEMENTATION_AGENT.md)
- **Review Agent:** Reviews tests and code, finds bugs (see REVIEW_AGENT.md)

---

## Coding Standards

### Python Style
- Type hints: Python 3.10+ style (`list[...]`, `str | None`, not `List`, `Optional`)
- Formatting: PEP 8 compliant
- Docstrings: Google style for all public methods
- Naming: `PascalCase` (classes), `snake_case` (functions), `UPPER_SNAKE_CASE` (constants)

### Type Annotation Examples
```python
# Correct
def get_legal_moves(self) -> list[chess.Move]:
    return list(self.board.legal_moves)

def process(self, game: ChessGame) -> tuple[np.ndarray, float]:
    return state, value

# Avoid
from typing import List, Optional  # Don't use
def old_style(self) -> Optional[List[chess.Move]]: pass  # Don't use
```

---

## Component Specifications

### 1. ChessGame Class (Game Environment)

See [specs/BOARD_REPRESENTATION.md](specs/BOARD_REPRESENTATION.md) for board tensor specification.

**Purpose:** Wrapper around python-chess providing AlphaZero-specific interfaces.

**Key Methods:**
```python
def __init__(self, fen: str | None = None)
def clone() -> ChessGame
def get_state() -> str
def get_legal_moves() -> list[chess.Move]
def get_legal_moves_mask() -> np.ndarray  # Shape (4672,)
def make_move(move: chess.Move) -> None
def undo_move() -> None
def is_game_over() -> bool
def get_result() -> float | None  # +1.0, 0.0, -1.0, None
def get_canonical_board() -> np.ndarray  # Shape (8, 8, 18)
def get_move_index(move: chess.Move) -> int  # Returns [0, 4671]
def get_move_from_index(index: int) -> chess.Move
```

### 2. ChessNN Class (Neural Network)

See [specs/NEURAL_NETWORK.md](specs/NEURAL_NETWORK.md) for full specification.

**Summary:** AlphaZero-style network with 10 residual blocks, 128 filters, dual policy/value heads.

### 3. MCTS Class (Monte Carlo Tree Search)

See [specs/MCTS.md](specs/MCTS.md) for full specification.

**Summary:** Tree search using neural network for evaluation and move priors.

### 4. Data Preparation

See [specs/DATA_PREPARATION.md](specs/DATA_PREPARATION.md) for full specification.

**Summary:** Download and process Lichess games with pre-computed Stockfish evaluations. Filter for balanced positions (±200 centipawns) to get high-quality training data.

### 5. Training Pipeline

See [specs/TRAINING_PIPELINE.md](specs/TRAINING_PIPELINE.md) for full specification.

**Phase 1: Supervised Pre-training**
- Data: Balanced positions from Lichess (with embedded Stockfish evals)
- Policy targets: One-hot encoding of the move played
- Value targets: Stockfish centipawn evaluation converted to [-1, 1]
- Expected result: ~1500-1700 ELO

**Stockfish Value Conversion:**
```python
def cp_to_value(cp: float, k: float = 400.0) -> float:
    """Convert centipawn to value in [-1, 1]."""
    if cp > 10000: return 1.0    # Mate
    if cp < -10000: return -1.0  # Mated
    return np.tanh(cp / k)
```

**Phase 2: Self-Play Refinement**
- Generate games using MCTS + current network
- Train on (position, MCTS policy, game outcome)
- Update network, repeat

---

## Testing Strategy

See [TEST_GUIDELINES.md](TEST_GUIDELINES.md) for detailed test naming conventions and organization.

### Test Categories
- Unit tests: Each component isolated
- Integration tests: Components working together
- Edge cases: Special moves, draw conditions
- Performance tests: Speed, memory usage

### Test Naming
```python
def test_<component>_<scenario>_<expected_behavior>():
    """Test that <component> <expected_behavior> when <scenario>."""
    pass
```

### Fixtures
- `fresh_game`: Standard starting position
- `endgame_position`: Simple endgame
- `checkmate_position`: Known checkmate
- `stalemate_position`: Known stalemate

---

## Common Patterns

### Error Handling
```python
def make_move(self, move: chess.Move) -> None:
    if move not in self.board.legal_moves:
        raise ValueError(f"Illegal move: {move.uci()}")
    self.board.push(move)
```

### Tensor Validation
```python
def predict(self, board: np.ndarray) -> tuple[np.ndarray, float]:
    assert board.shape == (8, 8, 18), f"Expected (8,8,18), got {board.shape}"
    # ...
```

### MCTS State Management
```python
def explore(self, game: ChessGame) -> ChessGame:
    child = game.clone()  # Independent copy
    child.make_move(best_move)
    return child
```

---

## Performance Notes

- Use float32 (not float64) for tensors
- Batch inference when possible
- Use numpy operations over Python loops
- tensorflow-metal provides M1 GPU acceleration
- Monitor memory usage (M1 Air thermal constraints)

---

## Data Sources

**Training Data:**
- Lichess Database: https://database.lichess.org/
- Format: PGN with embedded Stockfish evaluations
- Filter: Balanced positions (±200 cp), skip first 8 moves
- Parsing: Use `chess.pgn.read_game()` with streaming decompression

---

## References

- AlphaZero paper: https://arxiv.org/abs/1712.01815
- python-chess docs: https://python-chess.readthedocs.io/
- TensorFlow Metal: https://developer.apple.com/metal/tensorflow-plugin/
- Lichess Database: https://database.lichess.org/

---

## Questions for Agents

**Architecture Agent:** Component design, specifications, integration
**Implementation Agent:** See IMPLEMENTATION_AGENT.md for TDD workflow
**Review Agent:** See REVIEW_AGENT.md for review process

**Note:** Research experiments and optimizations are documented in EXPERIMENTS.md (separate track).
