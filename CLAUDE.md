# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# OmegaZero Chess Engine - Project Documentation

## Project Overview

OmegaZero is the name of an educational implementation of an AlphaZero-style
chess engine built from scratch to learn fundamentals of deep learning,
reinforcement learning, and Monte Carlo Tree Search (MCTS).

**Goals:**
- Understand AlphaZero architecture deeply through implementation
- Build working chess engine using self-play reinforcement learning
- Train locally on MacBook Air M1 GPU

**Learning Focus:**
- Neural network architectures (policy + value heads)
- Monte Carlo Tree Search algorithms
- Self-play training loops
- Reinforcement learning fundamentals

## Technical Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow with tensorflow-metal (M1 GPU support)
- **Chess Rules:** python-chess library (handles legal move generation)
- **Testing:** pytest
- **Type Checking:** Python type hints (3.10+ style)
- **Dependencies:** numpy, chess, tensorflow, tensorflow-metal

## Hardware Constraints

- **Platform:** MacBook Air M1
- **GPU:** Apple Silicon (via tensorflow-metal, not CUDA)
- **Memory:** Limited compared to original AlphaZero (5,000 TPUs)
- **Implications:**
  - Smaller neural networks
  - Fewer MCTS simulations per move
  - Longer training times
  - May need to start with simplified game variants

## Project Structure
```
alphazero-chess/
├── CLAUDE.md                    # This file
├── IMPLEMENTATION_AGENT.md      # Implementation Agent instructions
├── REVIEW_AGENT.md              # Review Agent instructions
├── README.md                    # Project overview and setup
├── requirements.txt             # Python dependencies
├── src/
│   ├── chess_game.py           # Game environment & rules engine
│   ├── neural_network.py       # Policy + value network
│   ├── mcts.py                 # Monte Carlo Tree Search
│   ├── self_play.py            # Self-play game generation
│   ├── training.py             # Training pipeline
│   └── evaluation.py           # Model evaluation & arena
├── tests/
│   ├── test_chess_game.py      # Game environment tests
│   ├── test_neural_network.py  # Neural network tests
│   ├── test_mcts.py            # MCTS tests
│   └── test_integration.py     # End-to-end tests
├── notebooks/
│   └── experiments/            # Jupyter notebooks for analysis
├── models/
│   └── checkpoints/            # Saved model weights
└── data/
    └── self_play_games/        # Generated training data
```

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_chess_game.py

# Run single test
pytest tests/test_chess_game.py::test_chess_game_castling_denied_after_king_moved

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src
```

## Current Status

**Phase:** Component 2 - Neural Network Architecture
**Status:** Planning complete, ready for test development
**Next:** Implementation Agent to write tests and implementation

**Completed Components:**
1. ✅ Game Environment & Rules Engine (chess_game.py)

**In Progress:**
2. 🔄 Neural Network Architecture (neural_network.py)

**Upcoming:**
3. ⏳ MCTS (mcts.py)
4. ⏳ Self-Play Engine (self_play.py)
5. ⏳ Training Pipeline (training.py)
6. ⏳ Evaluation System (evaluation.py)

## Development Approach

### Test-Driven Development (TDD)

1. **Write tests first** for each component
2. **Implement** to pass tests
3. **Refactor** while keeping tests green
4. **Iterate** component by component

### Three-Agent Development Workflow

**Architecture Agent (Planning):**
- Designs component architecture and specifications
- Makes high-level design decisions
- Provides detailed prompts for Implementation Agent

**Implementation Agent (Building):**
- Writes comprehensive tests FIRST (TDD)
- Implements components to pass tests
- Follows IMPLEMENTATION_AGENT.md guidelines

**Review Agent (Quality Assurance):**
- Reviews test coverage and completeness
- Checks for test overfitting
- Finds bugs and edge cases
- Suggests and implements improvements
- Follows REVIEW_AGENT.md guidelines

## Coding Standards

### Python Style

- **Type hints:** Always use Python 3.10+ style
  - ✅ `list[chess.Move]` not `List[chess.Move]`
  - ✅ `str | None` not `Optional[str]`
  - ✅ `dict[str, int]` not `Dict[str, int]`
- **Formatting:** PEP 8 compliant
- **Docstrings:** Google style for all public methods
- **Naming:**
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`

### Type Annotation Examples
```python
# Good - Python 3.10+ style
def get_legal_moves(self) -> list[chess.Move]:
    """Get all legal moves from current position."""
    return list(self.board.legal_moves)

def process_game(self, game: ChessGame) -> tuple[np.ndarray, float]:
    """Process game and return state and value."""
    state = game.get_canonical_board()
    value = game.get_result()
    return state, value

# Avoid - Old style
from typing import List, Optional, Tuple  # Don't import these
def old_style(self) -> Optional[List[chess.Move]]:  # Don't use
    pass
```

## Key Design Decisions

### ChessGame Class

**Philosophy:** Thin wrapper around python-chess
- **python-chess handles:** Legal moves, game rules, edge cases
- **Our wrapper provides:** AlphaZero-specific interfaces
- **Rationale:** Chess rules are complex; leverage mature library

**Critical methods:**
- `get_canonical_board()` - Always from current player's perspective
- `get_move_index()` / `get_move_from_index()` - Move ↔ policy index
- `clone()` - Independent state copies for MCTS tree exploration

### Move Encoding

**AlphaZero approach:** 73 planes × 64 squares = 4,672 possible moves
- 56 planes: Queen-style moves (8 directions × 7 distances)
- 8 planes: Knight moves
- 9 planes: Underpromotions (3 directions × 3 pieces)

**Alternative:** Simplified 64×64 = 4,096 encoding (from_square × to_square)
- Use simplified version initially if AlphaZero encoding too complex
- Can upgrade later without changing architecture

### Board Representation

**Input planes (14 total):**
- Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
- Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
- Plane 12: Repetition count (for draw detection)
- Plane 13: En passant square

**Canonical form:** Always flip board if black to move
- Reduces what network needs to learn
- "My pieces vs opponent's pieces" not "white vs black"

## Neural Network Architecture Details

**Architecture Type:** AlphaZero-style with residual blocks

**Network Structure:**
```
Input: (batch_size, 8, 8, 14)
    ↓
Initial Conv Block
    Conv2D: 128 filters, 3×3, padding='same'
    BatchNorm + ReLU
    ↓
Residual Tower (10 blocks)
    Each block:
    - Conv2D: 128 filters, 3×3, padding='same', use_bias=False
    - BatchNorm + ReLU
    - Conv2D: 128 filters, 3×3, padding='same', use_bias=False
    - BatchNorm
    - Add (skip connection)
    - ReLU
    ↓
    ┌──────────────┴──────────────┐
    ↓                             ↓
Policy Head                  Value Head
Conv2D: 2 filters, 1×1       Conv2D: 1 filter, 1×1
BatchNorm + ReLU             BatchNorm + ReLU
Flatten                      Flatten
Dense: 4672                  Dense: 256 + ReLU
Softmax                      Dense: 1 + Tanh
    ↓                             ↓
(batch, 4672)                (batch, 1)
```

**Model Configuration:**
- **Residual blocks:** 10 (scalable: 5 for debugging, 15-20 for stronger play)
- **Filters per layer:** 128 (scalable: 64 for faster training, 256 for stronger play)
- **Total parameters:** ~3.6M (manageable on M1 MacBook Air)
- **Batch normalization:** Yes, throughout (critical for deep networks)
- **Skip connections:** Yes, in residual blocks (enables deep learning)

**Why Residual Blocks:**
- Enable training deep networks (10+ layers) without gradient vanishing
- Critical innovation that makes AlphaZero architecture work
- Allows network to learn complex chess patterns
- Skip connections provide gradient highways during backpropagation

**Input/Output Specifications:**
- **Input shape:** `(batch_size, 8, 8, 14)` - board planes from ChessGame
- **Policy output:** `(batch_size, 4672)` - move probabilities, softmax activation
- **Value output:** `(batch_size, 1)` - position evaluation in [-1, 1], tanh activation
- **Data type:** float32 throughout (required for M1 GPU efficiency)

**Batch Normalization:**
- Used after every convolution (before activation)
- use_bias=False in Conv layers (BatchNorm makes bias redundant)
- Critical for training stability in deep networks
- **Training mode:** `model(x, training=True)` updates statistics
- **Inference mode:** `model(x, training=False)` uses frozen statistics

### MCTS Conventions

- **Visit counts:** Track as integers
- **Action values Q:** Track as floats in [-1, 1]
- **Prior probabilities P:** From policy head, in [0, 1]
- **PUCT constant:** Tunable hyperparameter (typically ~1.0)

## Training Strategy: Hybrid Approach

**Philosophy:** Combine supervised learning and self-play for practical training on M1 hardware.

### Phase 1: Supervised Pre-training (Recommended First Step)

**Data Source:** Grandmaster game databases
- Lichess Elite Database: https://database.lichess.org/
- 3+ million games from players rated 2200+
- ~100 million positions for training
- Free download, PGN format

**Training Procedure:**
1. Parse PGN files to extract positions
2. For each position:
   - Input: canonical board tensor (8, 8, 14)
   - Policy target: one-hot encoding of GM's move
   - Value target: game outcome (+1, 0, -1)
3. Train network on GM data (10-20 epochs)
4. Expected result: ~1500-1700 ELO in 1-2 days

**Benefits:**
- Fast bootstrap (hours vs. weeks)
- Learns proven opening theory
- Learns positional concepts from strong play
- Provides strong baseline for self-play

### Phase 2: Self-Play Refinement

**After pre-training, switch to AlphaZero self-play:**
1. Generate games using current network + MCTS
2. Train on self-play data (positions, MCTS policies, outcomes)
3. Network improves beyond human training data
4. Can discover novel strategies

**Advantages over pure self-play:**
- Start from competent baseline (~1600 ELO)
- Higher quality self-play games
- Faster convergence to strong play
- More efficient use of M1 compute resources

**This is what Leela Chess Zero does** - it's a proven, practical approach.

### Expected Performance Timeline (M1 MacBook Air)

**Supervised Pre-training:**
- Day 1-2: Process 100M GM positions
- Day 2-3: Train network on GM data
- Result: ~1500-1700 ELO baseline

**Self-Play Phase 1 (Week 1-2):**
- Generate: ~10,000 self-play games
- Result: ~1700-1900 ELO

**Self-Play Phase 2 (Week 3-8):**
- Generate: ~50,000-100,000 games
- Result: ~1900-2100 ELO

**Long-term (3+ months):**
- Plateau around 1800-2000 ELO on M1 alone
- Higher ELO requires more compute (cloud GPUs)

### Why Not Pure AlphaZero Self-Play?

**Pure self-play challenges:**
- Millions of terrible games before learning basics
- Reinvents basic chess knowledge from scratch
- Requires massive compute ($25M+ for original AlphaZero)
- Not practical for M1 MacBook Air

**Hybrid approach advantages:**
- Practical for limited compute
- Achieves same end goal (strong chess engine)
- More educational (learn both supervised and RL)
- Faster time to playable strength

## Testing Strategy

### Test Categories

1. **Unit tests:** Each component in isolation
2. **Integration tests:** Components working together
3. **Edge case tests:** Special moves, draw conditions
4. **Performance tests:** MCTS speed, memory usage

### Test Naming Convention
```python
def test_<component>_<scenario>_<expected_behavior>():
    """
    Test that <component> <expected_behavior> when <scenario>.
    """
    pass

# Examples:
def test_chess_game_castling_denied_after_king_moved():
    """Test that castling is illegal after king has moved."""

def test_mcts_exploration_balances_exploitation():
    """Test that MCTS explores unvisited nodes while exploiting good moves."""
```

### pytest Fixtures

Create fixtures for common setups:
- `fresh_game`: Standard starting position
- `endgame_position`: Simple endgame for quick testing
- `checkmate_position`: Known checkmate
- `stalemate_position`: Known stalemate

## Dependencies Management

### requirements.txt
```
numpy>=1.24.0
python-chess>=1.999
tensorflow>=2.15.0
tensorflow-metal>=1.1.0  # For M1 GPU
pytest>=7.4.0
```

## Common Patterns

### Error Handling
```python
# Always validate inputs
def make_move(self, move: chess.Move) -> None::
    if move not in self.board.legal_moves:
        raise ValueError(f"Illegal move: {move.uci()}")
    self.board.push(move)
```

### Tensor Shape Checking
```python
# Always validate tensor shapes
def predict(self, board: np.ndarray) -> tuple[np.ndarray, float]:
    assert board.shape == (8, 8, 14), f"Expected (8,8,14), got {board.shape}"
    # ... neural network forward pass
```

### MCTS State Management
```python
# Always clone before exploring branches
def select_child(self, game: ChessGame) -> tuple[ChessGame, chess.Move]:
    best_move = self._select_best_move()
    child_game = game.clone()  # Independent copy
    child_game.make_move(best_move)
    return child_game, best_move
```

## Performance Considerations

### Memory Optimization

- Use `float32` not `float64` for neural network tensors
- Limit self-play game buffer size
- Clear old model checkpoints periodically

### Computation Optimization

- Batch neural network inference when possible
- Use numpy operations over Python loops
- Profile MCTS to find bottlenecks

### M1-Specific

- tensorflow-metal provides GPU acceleration
- Expect modest speedups vs CPU (not CUDA-level)
- Monitor memory usage (M1 Air has thermal constraints)

## Training Data and Databases

**Grandmaster Games:**
- Lichess Elite Database: https://database.lichess.org/
- FICS Games Database
- Chess.com games (with account)

**PGN Parsing:**
- Use python-chess library: `chess.pgn.read_game()`
- Extract positions, moves, and outcomes
- Convert to training format (tensors + targets)

**Data Augmentation:**
- Horizontal flip for symmetry (optional, chess is mostly symmetric)
- Mix GM data with self-play data during transition phase

## Future Considerations

- **Simplification:** May start with Connect-4 or smaller chess variant to validate pipeline
- **Scaling:** If training too slow, consider smaller network or fewer simulations
- **Evaluation:** Track ELO ratings to measure improvement
- **Visualization:** TensorBoard for training metrics
- **Cloud Training:** Rent GPU for intensive training periods if needed

## Questions for Architecture Agent

When asking for architectural guidance:
1. Specify which component you're planning
2. Ask about design decisions and tradeoffs
3. Request component specifications and interfaces
4. Discuss integration between components
5. Get recommendations on implementation approaches

## Questions for Implementation Agent

When implementing a component:
1. Reference IMPLEMENTATION_AGENT.md for TDD workflow
2. Write comprehensive tests FIRST
3. Implement to pass tests
4. Follow all coding standards in CLAUDE.md
5. Use proper Python 3.10+ type hints

## Questions for Review Agent

When reviewing code:
1. Reference REVIEW_AGENT.md for review process
2. Check for test coverage gaps
3. Look for overfitting to tests
4. Find bugs and edge cases
5. Suggest improvements to tests and implementation

## References

- AlphaZero paper: https://arxiv.org/abs/1712.01815
- Leela Chess Zero: https://lczero.org/
- python-chess docs: https://python-chess.readthedocs.io/
- TensorFlow Metal: https://developer.apple.com/metal/tensorflow-plugin/
- Lichess Database: https://database.lichess.org/
