# TEST_GUIDELINES.md

# Test Naming and Organization Guidelines

**This document defines the systematic approach to test naming and organization for OmegaZero.**

Read this document thoroughly before writing any tests. Following these conventions ensures consistency across the codebase and makes test suites manageable as they grow to hundreds of tests.

-----

## Core Principles

For a comprehensive system like OmegaZero requiring extensive test coverage, use a **hybrid strategy**:

1. **Classes provide context** - Group tests by the component/function being tested
2. **Parametrization reduces duplication** - Use `@pytest.mark.parametrize` for input variations
3. **Category prefixes within classes** - Systematic prefixes distinguish test types
4. **IDs describe specific cases** - Parametrized test IDs replace verbose function names

-----

## Test Naming Convention

### Within Test Classes, Use Category Prefixes

| Prefix | Purpose | Example |
|--------|---------|---------|
| `test_valid_*` | Happy path, expected behavior | `test_valid_move_updates_board` |
| `test_edge_*` | Boundary conditions, limits | `test_edge_stalemate_has_no_moves` |
| `test_error_*` | Error handling, invalid inputs | `test_error_invalid_fen_raises_value_error` |
| `test_canonical_*` | NN representation specifics | `test_canonical_board_flips_for_black` |
| `test_integration_*` | Component interactions | `test_integration_game_with_mcts` |
| `test_regression_*` | Bug fixes (include issue ID) | `test_regression_gh42_castling_rights` |

### Class Naming

Name test classes as `Test<Component><Aspect>`:

```python
class TestChessGameInitialization: ...
class TestChessGameMoveExecution: ...
class TestChessGameLegalMoves: ...
class TestNeuralNetworkForwardPass: ...
class TestNeuralNetworkResidualBlocks: ...
class TestMCTSNodeExpansion: ...
class TestMCTSBackpropagation: ...
```

-----

## Parametrization Guidelines

### Always Use Descriptive IDs

```python
# ✅ Good: Descriptive IDs
@pytest.mark.parametrize('fen,expected', [
    pytest.param(STARTING_FEN, 20, id="starting_position"),
    pytest.param(ITALIAN_GAME_FEN, 33, id="italian_game"),
    pytest.param(ENDGAME_FEN, 9, id="king_pawn_endgame"),
])
def test_valid_legal_move_count(self, fen: str, expected: int) -> None: ...

# ❌ Bad: No IDs (pytest generates confusing names like "[fen0-20]")
@pytest.mark.parametrize('fen,expected', [
    (STARTING_FEN, 20),
    (ITALIAN_GAME_FEN, 33),
])
def test_legal_move_count(self, fen: str, expected: int) -> None: ...
```

### When to Parametrize

Use parametrization for:

- **Multiple board positions** (starting, midgame, endgame)
- **Move variations** (different piece types, directions)
- **Input variations** (valid/invalid FEN strings)
- **Expected outcomes** (move counts, piece placements)
- **Neural network inputs** (different batch sizes, board states)

### When NOT to Parametrize

Keep separate test functions when:

- Tests require significantly different setup
- Failure messages would be unclear
- Tests verify fundamentally different behaviors

-----

## Complete Example: Test File Structure

```python
# tests/test_chess_game.py
"""
Comprehensive tests for ChessGame component.

Test classes:
- TestChessGameInitialization
- TestChessGameMoveExecution
- TestChessGameLegalMoves
- TestChessGameStatus
- TestChessGameNeuralNetworkInterface
- TestChessGameSpecialMoves
"""

import pytest
import numpy as np
import chess
from src.chess_game import ChessGame


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def starting_position() -> ChessGame:
    """Standard starting position."""
    return ChessGame()


@pytest.fixture
def midgame_position() -> ChessGame:
    """A typical midgame position."""
    return ChessGame(fen="r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")


@pytest.fixture
def endgame_position() -> ChessGame:
    """A basic endgame position."""
    return ChessGame(fen="8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestChessGameInitialization:
    """Tests for ChessGame initialization and construction."""
    
    # -------------------------------------------------------------------------
    # Valid initialization tests
    # -------------------------------------------------------------------------
    
    def test_valid_default_creates_starting_position(self) -> None:
        """Default initialization creates standard starting position."""
        game = ChessGame()
        expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert game.get_state() == expected_fen
    
    @pytest.mark.parametrize('fen', [
        pytest.param(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            id="italian_game"
        ),
        pytest.param(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            id="kings_pawn_opening"
        ),
        pytest.param(
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            id="king_pawn_endgame"
        ),
    ])
    def test_valid_custom_fen_accepted(self, fen: str) -> None:
        """Custom FEN strings are parsed correctly."""
        game = ChessGame(fen=fen)
        assert game.get_state() == fen
    
    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------
    
    def test_edge_empty_fen_uses_default(self) -> None:
        """Empty string FEN falls back to starting position."""
        game = ChessGame(fen="")
        assert game.get_state() == chess.STARTING_FEN
    
    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.parametrize('invalid_fen', [
        pytest.param("not a valid fen", id="garbage_string"),
        pytest.param("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1", id="missing_rank"),
        pytest.param("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1", id="invalid_side"),
    ])
    def test_error_invalid_fen_raises_value_error(self, invalid_fen: str) -> None:
        """Invalid FEN strings raise ValueError."""
        with pytest.raises(ValueError):
            ChessGame(fen=invalid_fen)


class TestChessGameMoveExecution:
    """Tests for move execution via make_move()."""
    
    # -------------------------------------------------------------------------
    # Valid move tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.parametrize('move_uci,expected_piece_square', [
        pytest.param("e2e4", ("P", chess.E4), id="pawn_push"),
        pytest.param("g1f3", ("N", chess.F3), id="knight_development"),
        pytest.param("b1c3", ("N", chess.C3), id="queenside_knight"),
    ])
    def test_valid_move_updates_board(
        self,
        starting_position: ChessGame,
        move_uci: str,
        expected_piece_square: tuple[str, int]
    ) -> None:
        """Legal moves update board state correctly."""
        move = chess.Move.from_uci(move_uci)
        starting_position.make_move(move)
        
        piece_symbol, square = expected_piece_square
        board = starting_position.board
        assert board.piece_at(square).symbol() == piece_symbol
    
    def test_valid_move_changes_side_to_move(self, starting_position: ChessGame) -> None:
        """Making a move switches the side to move."""
        assert starting_position.board.turn == chess.WHITE
        starting_position.make_move(chess.Move.from_uci("e2e4"))
        assert starting_position.board.turn == chess.BLACK
    
    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.parametrize('illegal_uci', [
        pytest.param("e2e5", id="pawn_moves_too_far"),
        pytest.param("e1e2", id="king_blocked_by_pawn"),
        pytest.param("a1a5", id="rook_blocked_by_pawn"),
    ])
    def test_error_illegal_move_raises_value_error(
        self,
        starting_position: ChessGame,
        illegal_uci: str
    ) -> None:
        """Illegal moves raise ValueError."""
        move = chess.Move.from_uci(illegal_uci)
        with pytest.raises(ValueError, match="Illegal move"):
            starting_position.make_move(move)


class TestChessGameLegalMoves:
    """Tests for legal move generation."""
    
    def test_valid_starting_position_has_twenty_moves(
        self,
        starting_position: ChessGame
    ) -> None:
        """Starting position has exactly 20 legal moves."""
        legal_moves = starting_position.get_legal_moves()
        assert len(legal_moves) == 20
    
    @pytest.mark.parametrize('fen,expected_count', [
        pytest.param(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            33,
            id="italian_game_white"
        ),
        pytest.param(
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            9,
            id="simple_endgame"
        ),
    ])
    def test_valid_position_move_count(self, fen: str, expected_count: int) -> None:
        """Positions have correct number of legal moves."""
        game = ChessGame(fen=fen)
        assert len(game.get_legal_moves()) == expected_count
    
    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------
    
    def test_edge_stalemate_has_no_moves(self) -> None:
        """Stalemate position has zero legal moves."""
        game = ChessGame(fen="k7/8/1K6/8/8/8/8/8 b - - 0 1")
        assert len(game.get_legal_moves()) == 0
    
    def test_edge_checkmate_has_no_moves(self) -> None:
        """Checkmate position has zero legal moves."""
        game = ChessGame(fen="rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        assert len(game.get_legal_moves()) == 0


class TestChessGameNeuralNetworkInterface:
    """Tests for neural network input/output interface."""
    
    def test_valid_canonical_board_shape(self, starting_position: ChessGame) -> None:
        """Canonical board has correct shape (8, 8, 14)."""
        board = starting_position.get_canonical_board()
        assert board.shape == (8, 8, 14)
    
    def test_valid_canonical_board_dtype(self, starting_position: ChessGame) -> None:
        """Canonical board has float32 dtype."""
        board = starting_position.get_canonical_board()
        assert board.dtype == np.float32
    
    def test_canonical_board_flips_for_black(self) -> None:
        """Board is flipped when black to move (canonical representation)."""
        white_to_move = ChessGame(fen="8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")
        black_to_move = ChessGame(fen="8/8/4k3/8/8/4K3/4P3/8 b - - 0 1")
        
        white_board = white_to_move.get_canonical_board()
        black_board = black_to_move.get_canonical_board()
        
        assert not np.array_equal(white_board, black_board)


class TestChessGameSpecialMoves:
    """Tests for castling, en passant, and promotion."""
    
    # -------------------------------------------------------------------------
    # Castling tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.parametrize('fen,castle_uci,rook_dest', [
        pytest.param(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "e1g1",
            chess.F1,
            id="white_kingside"
        ),
        pytest.param(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "e1c1",
            chess.D1,
            id="white_queenside"
        ),
    ])
    def test_valid_castling_moves_rook(
        self,
        fen: str,
        castle_uci: str,
        rook_dest: int
    ) -> None:
        """Castling moves both king and rook."""
        game = ChessGame(fen=fen)
        game.make_move(chess.Move.from_uci(castle_uci))
        assert game.board.piece_at(rook_dest).piece_type == chess.ROOK
    
    # -------------------------------------------------------------------------
    # En passant tests
    # -------------------------------------------------------------------------
    
    def test_valid_en_passant_captures_pawn(self) -> None:
        """En passant capture removes the captured pawn."""
        game = ChessGame(fen="rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        game.make_move(chess.Move.from_uci("e5f6"))
        
        assert game.board.piece_at(chess.F5) is None
        assert game.board.piece_at(chess.F6).symbol() == "P"
    
    # -------------------------------------------------------------------------
    # Promotion tests
    # -------------------------------------------------------------------------
    
    @pytest.mark.parametrize('promo_piece', [
        pytest.param(chess.QUEEN, id="promote_queen"),
        pytest.param(chess.ROOK, id="promote_rook"),
        pytest.param(chess.BISHOP, id="promote_bishop"),
        pytest.param(chess.KNIGHT, id="promote_knight"),
    ])
    def test_valid_promotion_creates_piece(self, promo_piece: int) -> None:
        """Pawn promotion creates the specified piece."""
        game = ChessGame(fen="8/P7/8/8/8/8/8/4K2k w - - 0 1")
        move = chess.Move(chess.A7, chess.A8, promotion=promo_piece)
        game.make_move(move)
        assert game.board.piece_at(chess.A8).piece_type == promo_piece
```

-----

## Anti-Patterns to Avoid

### ❌ Ad-Hoc Test Names

```python
# BAD: Inconsistent, hard to categorize
def test_it_works(self) -> None: ...
def test_move_stuff(self) -> None: ...
def test_that_invalid_fen_fails(self) -> None: ...
def test_fen_validation_error(self) -> None: ...
```

### ✅ Systematic Names

```python
# GOOD: Consistent prefixes, clear categories
def test_valid_move_updates_board(self) -> None: ...
def test_valid_move_changes_side(self) -> None: ...
def test_error_invalid_fen_raises_value_error(self) -> None: ...
def test_error_illegal_move_raises_value_error(self) -> None: ...
```

### ❌ Duplicated Test Logic

```python
# BAD: Repetitive test functions
def test_legal_moves_starting_position(self) -> None:
    game = ChessGame(fen=STARTING_FEN)
    assert len(game.get_legal_moves()) == 20

def test_legal_moves_italian_game(self) -> None:
    game = ChessGame(fen=ITALIAN_GAME_FEN)
    assert len(game.get_legal_moves()) == 33

def test_legal_moves_endgame(self) -> None:
    game = ChessGame(fen=ENDGAME_FEN)
    assert len(game.get_legal_moves()) == 9
```

### ✅ Parametrized Tests

```python
# GOOD: Single parametrized test
@pytest.mark.parametrize('fen,expected_count', [
    pytest.param(STARTING_FEN, 20, id="starting_position"),
    pytest.param(ITALIAN_GAME_FEN, 33, id="italian_game"),
    pytest.param(ENDGAME_FEN, 9, id="simple_endgame"),
])
def test_valid_position_move_count(self, fen: str, expected_count: int) -> None:
    game = ChessGame(fen=fen)
    assert len(game.get_legal_moves()) == expected_count
```

-----

## Running Tests by Category

The systematic naming enables powerful test filtering:

```bash
# Run all tests for a component
PYTHONPATH=. pytest tests/test_chess_game.py -v

# Run specific test class
PYTHONPATH=. pytest tests/test_chess_game.py::TestChessGameInitialization -v

# Run specific test method
PYTHONPATH=. pytest tests/test_chess_game.py::TestChessGameInitialization::test_valid_default_creates_starting_position -v

# Run all "valid" (happy path) tests
PYTHONPATH=. pytest tests/test_chess_game.py -k "test_valid" -v

# Run all edge case tests
PYTHONPATH=. pytest tests/test_chess_game.py -k "test_edge" -v

# Run all error handling tests
PYTHONPATH=. pytest tests/test_chess_game.py -k "test_error" -v

# Combine filters
PYTHONPATH=. pytest tests/test_chess_game.py -k "test_valid or test_edge" -v

# Show test names only (review organization)
PYTHONPATH=. pytest tests/test_chess_game.py --collect-only
```

-----

## Test Coverage Checklist

For each component, ensure test classes cover:

✅ **Initialization (`TestXxxInitialization`)**

- Default initialization
- Custom initialization with parameters
- Edge cases (empty, None, boundaries)
- Error handling (invalid inputs)

✅ **Core Functionality (`TestXxxCoreBehavior`)**

- All public methods tested
- Return values correct (type and value)
- State changes handled correctly

✅ **Edge Cases (`test_edge_*` within relevant classes)**

- Boundary conditions (max batch size, empty inputs)
- Chess-specific edges (stalemate, checkmate, insufficient material)
- Numerical edges (zero values, maximum values)

✅ **Error Handling (`test_error_*` within relevant classes)**

- Invalid inputs raise appropriate exceptions
- Exception messages are informative
- No silent failures

✅ **Neural Network Interface (`TestXxxNeuralNetworkInterface`)**

- Tensor shapes match architecture requirements
- Data types are correct (float32)
- Canonical representation is consistent

-----

## Summary

Following these conventions ensures:

1. **Consistency** - All tests follow the same patterns
2. **Discoverability** - Easy to find tests for specific behaviors
3. **Maintainability** - Clear organization as test suite grows
4. **Filterability** - Run subsets of tests by category
5. **Readability** - Test names document expected behavior
