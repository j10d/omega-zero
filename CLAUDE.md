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
├── BOARD_REPRESENTATION.md      # Board tensor specification
├── NEURAL_NETWORK.md            # Neural network specification
├── EXPERIMENTS.md               # Research experiments (separate track)
├── src/
│   ├── chess_game.py           # Game environment & rules engine
│   ├── neural_network.py       # Policy + value network
│   ├── mcts.py                 # Monte Carlo Tree Search
│   ├── self_play.py            # Self-play game generation
│   ├── training.py             # Training pipeline
│   └── evaluation.py           # Model evaluation & arena
├── tests/
│   ├── test_chess_game.py
│   ├── test_neural_network.py
│   ├── test_mcts.py
│   └── test_integration.py
└── data/
    └── self_play_games/
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
3. ⏳ MCTS (mcts.py)
4. ⏳ Self-Play Engine (self_play.py)
5. ⏳ Training Pipeline (training.py)
6. ⏳ Evaluation System (evaluation.py)

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

See [BOARD_REPRESENTATION.md](BOARD_REPRESENTATION.md) for board tensor specification.

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

See [NEURAL_NETWORK.md](NEURAL_NETWORK.md) for full specification.

**Summary:** AlphaZero-style network with 10 residual blocks, 128 filters, dual policy/value heads.

### 3. MCTS Class (Monte Carlo Tree Search)

**Purpose:** Tree search using neural network for evaluation and move priors.

**PUCT Formula:**
```
PUCT(node) = Q(node) + c_puct * P(node) * sqrt(N_parent) / (1 + N(node))
```

**Key Components:**
- Visit counts (integers)
- Action values Q (floats in [-1, 1])
- Prior probabilities P (from policy head, in [0, 1])
- c_puct constant (typically ~1.0)

### 4. Training Pipeline

**Phase 1: Supervised Pre-training**
- Data: Lichess Elite Database (GM games, 2200+ rated)
- Policy targets: One-hot encoding of GM moves
- Value targets: Stockfish centipawn evaluations converted to [-1, 1]
- Training: 10-20 epochs on ~100M positions
- Expected result: ~1500-1700 ELO

**Stockfish Value Conversion:**
```python
def cp_to_value(cp: float, k: float) -> float:
    """Convert centipawn to value in [-1, 1]."""
    if cp > 10000: return 1.0    # Mate
    if cp < -10000: return -1.0  # Mated
    return np.tanh(cp / k)
```

**k Schedule:**
- Days 1-2: k=500
- Day 3+: k=400

**Phase 2: Self-Play Refinement**
- Generate games using MCTS + current network
- Train on (position, MCTS policy, game outcome)
- Update network, repeat

---

## Testing Strategy

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

**GM Games:**
- Lichess Elite Database: https://database.lichess.org/
- Format: PGN
- Parsing: Use `chess.pgn.read_game()`

**Position Evaluation:**
- Stockfish depth 15-20
- Convert centipawns to value targets using k parameter

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
