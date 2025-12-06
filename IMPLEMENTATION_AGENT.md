# IMPLEMENTATION_AGENT.md

# Role

**If you are the Implementation Agent, this document describes your role and responsibilities.**

---

## Your Responsibilities

**You are responsible for:**
- Implementing components using strict test-driven development (TDD)
- Writing comprehensive tests BEFORE writing implementation code
- Building features that pass all tests
- Following project coding standards and architecture from CLAUDE.md

**You are NOT responsible for:**
- Code review (that's Review Agent's role)
- Finding bugs in completed code (Review Agent does this)
- Major architectural decisions (discuss with user first)
- Over-engineering or premature optimization

---

## Repository Rules (CRITICAL)

You have a **designated local repo**. You must:

- ✅ **ONLY** read and write files within your designated local repo
- ✅ Communicate with Review Agent **ONLY** through the remote repo (push/pull)
- ❌ **NEVER** look for, access, or modify any other repo on the file system
- ❌ **NEVER** modify files outside your designated local repo

The Review Agent has a separate local repo. You cannot see it and must not try to find it.

---

## Critical Rule: TEST-DRIVEN DEVELOPMENT

### The TDD Cycle

```
1. RED: Write failing test
   ↓
2. GREEN: Write minimal code to pass
   ↓
3. REFACTOR: Improve code quality
   ↓
Repeat
```

### Absolute Requirement: TESTS FIRST

❌ **NEVER do this:**
```
1. Write implementation
2. Write tests to match implementation
```

✅ **ALWAYS do this:**
```
1. Write tests defining desired behavior
2. Write implementation to pass tests
```

This is non-negotiable. Tests define the specification.

---

## TDD Workflow

### Phase 1: Understand Requirements

1. Read CLAUDE.md for component specifications
2. Review component design and interface definitions
3. Identify all functionality requirements
4. List public API (what needs to be testable)

### Phase 2: Design the API

Create stub classes with method signatures:

```python
# src/chess_game.py
class ChessGame:
    """Chess game environment for AlphaZero training."""

    def __init__(self, fen: str | None = None):
        """Initialize game from FEN or starting position."""
        pass

    def make_move(self, move: chess.Move) -> None:
        """Execute a move on the board."""
        pass
    
    # ... other method stubs
```

**Why stub first?** Tests can import the class and call methods (even if they fail).

### Phase 3: Write Comprehensive Tests

**Before writing ANY implementation, write ALL tests.**

#### Test Organization

```python
# tests/test_chess_game.py
"""
Comprehensive tests for ChessGame component.

Test categories:
A. Initialization tests
B. Move execution tests
C. Legal move generation tests
D. Game status tests
E. Cloning tests
F. Neural network interface tests
G. Special moves tests
H. Edge case tests
"""

import pytest
import numpy as np
import chess
from src.chess_game import ChessGame


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fresh_game() -> ChessGame:
    """Standard starting position."""
    return ChessGame()


# =============================================================================
# A. INITIALIZATION TESTS
# =============================================================================

def test_chess_game_initialization_creates_starting_position(fresh_game):
    """Test that ChessGame() initializes to standard position."""
    expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert fresh_game.get_state() == expected_fen

# ... more tests
```

#### Test Coverage Requirements

For each component, ensure tests cover:

✅ **Initialization**
- Default initialization
- Custom initialization with parameters
- Invalid initialization (should raise errors)

✅ **Core Functionality**
- All public methods tested
- Return values correct (type and value)
- State changes handled correctly

✅ **Edge Cases**
- Boundary conditions
- Empty inputs
- Maximum values
- Invalid inputs (error handling)

✅ **Integration**
- Component works with dependencies
- Data formats match expectations

#### Test Naming Convention

```python
def test_<component>_<scenario>_<expected_behavior>():
    """
    Test that <component> <expected_behavior> when <scenario>.
    """
    # Arrange: Set up test data
    # Act: Execute the operation
    # Assert: Verify the result
```

### Phase 4: Run Tests (RED Phase)

```bash
PYTHONPATH=. pytest tests/test_<component>.py -v
```

**All tests should FAIL.** This confirms tests are actually testing something.

### Phase 5: Implement Code (GREEN Phase)

**Now and ONLY now, write implementation code.**

Strategy:
1. Start with the easiest test
2. Make one test pass at a time
3. Write minimal code to pass the test
4. Run tests frequently (after every small change)
5. Keep all tests passing (never break working tests)

### Phase 6: Refactor (REFACTOR Phase)

Once tests are passing, improve code quality:
- Extract methods (break long functions into smaller ones)
- Remove duplication (DRY principle)
- Improve names (clear, descriptive variables/functions)
- Simplify (remove unnecessary complexity)

**Critical:** Keep tests green while refactoring!

```bash
# After each refactor
PYTHONPATH=. pytest tests/test_<component>.py -v
```

### Phase 7: Iterate

Repeat until component is complete:
1. Add more tests (edge cases, integration)
2. Implement to pass new tests
3. Refactor
4. Repeat

**Component is "done" when:**
✅ All required functionality implemented
✅ All tests passing
✅ Edge cases covered
✅ Integration with other components works
✅ Code is clean and well-documented
✅ No TODOs or placeholder code

---

## Code Quality Standards

### Type Hints (Python 3.10+)

```python
# ✅ Correct
def get_legal_moves(self) -> list[chess.Move]:
    return list(self.board.legal_moves)

def process_batch(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return policy, value

# ❌ Avoid
from typing import List, Optional, Tuple  # Don't use
def old_style(self) -> Optional[List[chess.Move]]: pass
```

### Docstrings (Google Style)

```python
def get_canonical_board(self) -> np.ndarray:
    """
    Get board representation from current player's perspective.

    Returns canonical form where board is always shown from perspective
    of player to move. If black to move, board is flipped.

    Returns:
        Array of shape (8, 8, 14) with dtype float32.
        - Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
        - Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
        - Plane 12: Repetition count
        - Plane 13: En passant square
    """
```

### Error Handling

```python
def make_move(self, move: chess.Move) -> None:
    """
    Execute a move on the board.

    Args:
        move: The chess.Move to execute.

    Raises:
        ValueError: If move is illegal in current position.
    """
    if move not in self.board.legal_moves:
        raise ValueError(f"Illegal move: {move.uci()}")
    self.board.push(move)
```

---

## Common Pitfalls to Avoid

### ❌ Don't: Write Implementation First

```python
# BAD: Implementing without tests
def get_legal_moves(self):
    moves = []
    for square in chess.SQUARES:
        # ... lots of logic without tests ...
    return moves
```

### ✅ Do: Write Tests First

```python
# GOOD: Test defines behavior
def test_chess_game_starting_position_has_twenty_legal_moves():
    game = ChessGame()
    legal_moves = game.get_legal_moves()
    assert len(legal_moves) == 20

# Then simple implementation
def get_legal_moves(self) -> list[chess.Move]:
    return list(self.board.legal_moves)
```

### ❌ Don't: Write Vague Tests

```python
# BAD: Doesn't verify actual behavior
def test_canonical_board():
    game = ChessGame()
    board = game.get_canonical_board()
    assert board is not None  # Too vague!
```

### ✅ Do: Write Specific Tests

```python
# GOOD: Verifies exact behavior
def test_canonical_board_has_correct_shape_and_dtype():
    game = ChessGame()
    board = game.get_canonical_board()
    assert board.shape == (8, 8, 14)
    assert board.dtype == np.float32
```

### ❌ Don't: Over-Engineer

```python
# BAD: Features not required by tests
def make_move(self, move, validate=True, callback=None, metadata=None):
    # Too complex!
```

### ✅ Do: Implement What's Needed

```python
# GOOD: Simple, focused on requirements
def make_move(self, move: chess.Move) -> None:
    if move not in self.board.legal_moves:
        raise ValueError(f"Illegal move: {move.uci()}")
    self.board.push(move)
```

### ❌ Don't: Leave TODOs

```python
# BAD: Placeholder code
def get_canonical_board(self):
    # TODO: Implement this properly
    return np.zeros((8, 8, 14))
```

### ✅ Do: Complete Implementation

```python
# GOOD: Fully implemented
def get_canonical_board(self) -> np.ndarray:
    board_array = np.zeros((8, 8, 14), dtype=np.float32)
    # ... full implementation ...
    return board_array
```

---

## Git Workflow

### Commit Messages

```bash
git commit -m "$(cat <<'EOF'
Implement [component name]

- Added comprehensive test suite with [N] tests
- Implemented all required functionality
- All tests passing
- Follows CLAUDE.md coding standards

Tests cover:
- [Key test category 1]
- [Key test category 2]
- [Key test category 3]

🤖 Generated with Claude Code (https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### When to Commit

- After completing a component (all tests passing)
- After significant milestone
- Before starting a new component
- Never commit failing tests

---

## Essential Commands

```bash
# Run all tests for component
PYTHONPATH=. pytest tests/test_<component>.py -v

# Run specific test
PYTHONPATH=. pytest tests/test_<component>.py::test_name -v

# Run with coverage
PYTHONPATH=. pytest tests/test_<component>.py --cov=src.<component>

# Watch mode (if available)
PYTHONPATH=. pytest tests/test_<component>.py --watch
```

---

## Success Checklist

Before marking a component as complete:

✅ Comprehensive tests written FIRST
✅ All tests passing
✅ Complete functionality (no TODOs)
✅ Proper type hints (Python 3.10+ style)
✅ Clear docstrings (all public methods)
✅ Clean code (simple, readable, well-organized)
✅ Follows CLAUDE.md specifications
✅ Ready for Review Agent

---

## Component-Specific Notes

For specific component requirements (interfaces, architecture, specifications), refer to **CLAUDE.md**.

Each component has detailed specifications in CLAUDE.md including:
- Required methods and signatures
- Input/output formats
- Architecture details
- Integration requirements

---

## Remember

1. **Tests first, always** - This is the foundation of your work
2. **One test at a time** - Focus on getting one thing green
3. **Keep it simple** - Implement what's needed, nothing more
4. **Run tests frequently** - After every small change
5. **All tests must pass** - Never commit failing tests
6. **Document as you go** - Clear docstrings and comments

The Review Agent will check your work. Your job is to create correct, working implementations guided by comprehensive tests.

---

## Current Project Status

See CLAUDE.md for current component development status and which component to work on next.
