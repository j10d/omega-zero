"""
Comprehensive unit tests for ChessGame class (Game Environment & Rules Engine).

This test suite follows TDD principles and covers:
- Initialization and state management
- Move execution and validation
- Legal move generation and masking
- Game status detection (checkmate, stalemate, draws)
- State cloning and independence
- Neural network interface (canonical board, move encoding)
- Special moves (castling, en passant, promotion)
- Edge cases and complex scenarios

Test organization:
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
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture
def fresh_game() -> ChessGame:
    """Standard starting position for chess."""
    return ChessGame()


@pytest.fixture
def endgame_position() -> ChessGame:
    """Simple K+Q vs K endgame for quick testing."""
    fen = "7k/8/8/8/8/8/4Q3/4K3 w - - 0 1"
    return ChessGame(fen=fen)


@pytest.fixture
def stalemate_position() -> ChessGame:
    """Known stalemate position - white to move, no legal moves, not in check."""
    fen = "7k/5Q2/5K2/8/8/8/8/8 b - - 0 1"
    return ChessGame(fen=fen)


@pytest.fixture
def checkmate_position() -> ChessGame:
    """Known checkmate position - black is checkmated."""
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    return ChessGame(fen=fen)


@pytest.fixture
def castling_test_position() -> ChessGame:
    """Position where both sides can castle both ways."""
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    return ChessGame(fen=fen)


@pytest.fixture
def en_passant_position() -> ChessGame:
    """Position ready for en passant capture."""
    fen = "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1"
    return ChessGame(fen=fen)


@pytest.fixture
def promotion_position() -> ChessGame:
    """White pawn ready to promote."""
    fen = "8/P7/8/8/8/8/8/K6k w - - 0 1"
    return ChessGame(fen=fen)


# =============================================================================
# A. INITIALIZATION TESTS
# =============================================================================

def test_chess_game_initialization_creates_starting_position(fresh_game: ChessGame):
    """Test that ChessGame() initializes to standard starting position."""
    expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert fresh_game.get_state() == expected_fen


def test_chess_game_initialization_accepts_custom_fen():
    """Test that ChessGame can be initialized with custom FEN string."""
    # Position after c2-c4, allowing black's d4 pawn to capture en passant on c3
    custom_fen = "rnbqkbnr/pppp1ppp/8/8/2PpP3/8/PP3PPP/RNBQKBNR b KQkq c3 0 1"
    game = ChessGame(fen=custom_fen)
    assert game.get_state() == custom_fen


def test_chess_game_initialization_with_none_fen_creates_starting_position():
    """Test that ChessGame(fen=None) creates starting position."""
    game = ChessGame(fen=None)
    expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert game.get_state() == expected_fen


def test_chess_game_fen_roundtrip_preserves_state():
    """Test that exporting and re-importing FEN preserves game state."""
    original_fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    game1 = ChessGame(fen=original_fen)
    exported_fen = game1.get_state()
    game2 = ChessGame(fen=exported_fen)
    assert game2.get_state() == original_fen


# =============================================================================
# B. MOVE EXECUTION TESTS
# =============================================================================

def test_chess_game_make_move_updates_state_correctly(fresh_game: ChessGame):
    """Test that making a legal move updates the game state."""
    initial_fen = fresh_game.get_state()
    move = chess.Move.from_uci("e2e4")
    fresh_game.make_move(move)
    new_fen = fresh_game.get_state()
    assert new_fen != initial_fen
    assert "e4" in new_fen or "4P3" in new_fen  # Pawn on e4


def test_chess_game_make_move_raises_error_for_illegal_move(fresh_game: ChessGame):
    """Test that making an illegal move raises ValueError."""
    illegal_move = chess.Move.from_uci("e2e5")  # Pawn can't move 3 squares
    with pytest.raises(ValueError, match="Illegal move"):
        fresh_game.make_move(illegal_move)


def test_chess_game_undo_move_restores_previous_state(fresh_game: ChessGame):
    """Test that undo_move() restores the state before the last move."""
    initial_fen = fresh_game.get_state()
    move = chess.Move.from_uci("e2e4")
    fresh_game.make_move(move)
    fresh_game.undo_move()
    assert fresh_game.get_state() == initial_fen


def test_chess_game_multiple_moves_and_undos_preserve_state(fresh_game: ChessGame):
    """Test that multiple moves and undos work correctly."""
    state0 = fresh_game.get_state()

    fresh_game.make_move(chess.Move.from_uci("e2e4"))
    state1 = fresh_game.get_state()

    fresh_game.make_move(chess.Move.from_uci("e7e5"))
    state2 = fresh_game.get_state()

    fresh_game.make_move(chess.Move.from_uci("g1f3"))

    fresh_game.undo_move()
    assert fresh_game.get_state() == state2

    fresh_game.undo_move()
    assert fresh_game.get_state() == state1

    fresh_game.undo_move()
    assert fresh_game.get_state() == state0


def test_chess_game_move_history_tracking(fresh_game: ChessGame):
    """Test that move history is tracked correctly."""
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]

    for move in moves:
        fresh_game.make_move(move)

    # After 3 moves, should be able to undo 3 times
    fresh_game.undo_move()
    fresh_game.undo_move()
    fresh_game.undo_move()

    # Should be back to starting position
    expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert fresh_game.get_state() == expected_fen


# =============================================================================
# C. LEGAL MOVE GENERATION TESTS
# =============================================================================

def test_chess_game_starting_position_has_twenty_legal_moves(fresh_game: ChessGame):
    """Test that starting position has exactly 20 legal moves."""
    legal_moves = fresh_game.get_legal_moves()
    assert len(legal_moves) == 20


def test_chess_game_get_legal_moves_returns_list_of_chess_moves(fresh_game: ChessGame):
    """Test that get_legal_moves() returns list[chess.Move]."""
    legal_moves = fresh_game.get_legal_moves()
    assert isinstance(legal_moves, list)
    assert all(isinstance(move, chess.Move) for move in legal_moves)


def test_chess_game_legal_moves_mask_has_correct_shape(fresh_game: ChessGame):
    """Test that legal_moves_mask has shape (4672,)."""
    mask = fresh_game.get_legal_moves_mask()
    assert mask.shape == (4672,)
    assert mask.dtype == bool or mask.dtype == np.bool_


def test_chess_game_legal_moves_mask_matches_legal_moves(fresh_game: ChessGame):
    """Test that legal_moves_mask True values correspond to actual legal moves."""
    legal_moves = fresh_game.get_legal_moves()
    mask = fresh_game.get_legal_moves_mask()

    # Count of True values in mask should equal number of legal moves
    assert np.sum(mask) == len(legal_moves)

    # Each legal move should have corresponding True in mask
    for move in legal_moves:
        move_index = fresh_game.get_move_index(move)
        assert mask[move_index], f"Move {move.uci()} at index {move_index} should be True in mask"


def test_chess_game_legal_move_count_consistency(endgame_position: ChessGame):
    """Test that legal move count is consistent across different methods."""
    legal_moves = endgame_position.get_legal_moves()
    mask = endgame_position.get_legal_moves_mask()
    assert len(legal_moves) == np.sum(mask)


def test_chess_game_checkmate_position_has_zero_legal_moves(checkmate_position: ChessGame):
    """Test that checkmate position has no legal moves."""
    legal_moves = checkmate_position.get_legal_moves()
    assert len(legal_moves) == 0


# =============================================================================
# D. GAME STATUS TESTS
# =============================================================================

def test_chess_game_starting_position_is_not_game_over(fresh_game: ChessGame):
    """Test that starting position is not game over."""
    assert not fresh_game.is_game_over()


def test_chess_game_checkmate_is_detected(checkmate_position: ChessGame):
    """Test that checkmate is correctly detected."""
    assert checkmate_position.is_game_over()


def test_chess_game_stalemate_is_detected(stalemate_position: ChessGame):
    """Test that stalemate is correctly detected."""
    assert stalemate_position.is_game_over()
    assert stalemate_position.is_draw()


def test_chess_game_get_result_returns_none_when_game_not_over(fresh_game: ChessGame):
    """Test that get_result() returns None when game is not over."""
    assert fresh_game.get_result() is None


def test_chess_game_get_result_checkmate_from_winner_perspective():
    """Test that get_result() returns +1.0 for winner (black wins, black to move)."""
    # Fool's mate: white is checkmated, black to move
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    game = ChessGame(fen=fen)

    # White is checkmated, it's white's turn (but they lost)
    # From white's perspective (current player), result should be -1.0 (loss)
    result = game.get_result()
    assert result == -1.0


def test_chess_game_get_result_stalemate_returns_zero(stalemate_position: ChessGame):
    """Test that get_result() returns 0.0 for stalemate (draw)."""
    result = stalemate_position.get_result()
    assert result == 0.0


def test_chess_game_insufficient_material_king_vs_king():
    """Test that K vs K is detected as draw."""
    fen = "7k/8/8/8/8/8/8/K7 w - - 0 1"
    game = ChessGame(fen=fen)
    assert game.is_draw()


def test_chess_game_insufficient_material_king_knight_vs_king():
    """Test that K+N vs K is detected as draw."""
    fen = "7k/8/8/8/8/8/4N3/K7 w - - 0 1"
    game = ChessGame(fen=fen)
    assert game.is_draw()


def test_chess_game_insufficient_material_king_bishop_vs_king():
    """Test that K+B vs K is detected as draw."""
    fen = "7k/8/8/8/8/8/4B3/K7 w - - 0 1"
    game = ChessGame(fen=fen)
    assert game.is_draw()


def test_chess_game_fools_mate_sequence():
    """Test Fool's Mate sequence: f3, e6, g4, Qh4# - fastest checkmate."""
    game = ChessGame()

    # White plays f3
    game.make_move(chess.Move.from_uci("f2f3"))
    assert not game.is_game_over()

    # Black plays e6
    game.make_move(chess.Move.from_uci("e7e6"))
    assert not game.is_game_over()

    # White plays g4
    game.make_move(chess.Move.from_uci("g2g4"))
    assert not game.is_game_over()

    # Black plays Qh4# - checkmate
    game.make_move(chess.Move.from_uci("d8h4"))
    assert game.is_game_over()

    # White has no legal moves
    assert len(game.get_legal_moves()) == 0

    # White lost (it's white's turn but they're checkmated)
    assert game.get_result() == -1.0


def test_chess_game_scholars_mate_sequence():
    """Test Scholar's Mate sequence: e4, e5, Bc4, Nc6, Qh5, Nf6, Qxf7#."""
    game = ChessGame()

    moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]

    for move_uci in moves[:-1]:
        game.make_move(chess.Move.from_uci(move_uci))
        assert not game.is_game_over()

    # Final move is checkmate
    game.make_move(chess.Move.from_uci(moves[-1]))
    assert game.is_game_over()
    assert game.get_result() == -1.0  # Black lost


# =============================================================================
# E. CLONING TESTS
# =============================================================================

def test_chess_game_clone_creates_independent_copy(fresh_game: ChessGame):
    """Test that clone() creates an independent copy."""
    clone = fresh_game.clone()

    # Make move on original
    fresh_game.make_move(chess.Move.from_uci("e2e4"))

    # Clone should be unchanged
    expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert clone.get_state() == expected_fen


def test_chess_game_clone_preserves_all_state(castling_test_position: ChessGame):
    """Test that clone preserves position, turn, castling rights, en passant."""
    original_fen = castling_test_position.get_state()
    clone = castling_test_position.clone()
    assert clone.get_state() == original_fen


def test_chess_game_modifying_clone_does_not_affect_original(fresh_game: ChessGame):
    """Test that modifying clone doesn't affect original game."""
    original_state = fresh_game.get_state()
    clone = fresh_game.clone()

    # Make several moves on clone
    clone.make_move(chess.Move.from_uci("e2e4"))
    clone.make_move(chess.Move.from_uci("e7e5"))
    clone.make_move(chess.Move.from_uci("g1f3"))

    # Original should be unchanged
    assert fresh_game.get_state() == original_state


def test_chess_game_clone_of_clone_works_correctly(fresh_game: ChessGame):
    """Test that cloning a clone works correctly."""
    clone1 = fresh_game.clone()
    clone1.make_move(chess.Move.from_uci("e2e4"))

    clone2 = clone1.clone()
    clone2.make_move(chess.Move.from_uci("e7e5"))

    # All three should have different states
    assert fresh_game.get_state() != clone1.get_state()
    assert clone1.get_state() != clone2.get_state()
    assert fresh_game.get_state() != clone2.get_state()


# =============================================================================
# F. NEURAL NETWORK INTERFACE TESTS
# =============================================================================

def test_chess_game_canonical_board_has_correct_shape(fresh_game: ChessGame):
    """Test that get_canonical_board() returns shape (8, 8, 14)."""
    board = fresh_game.get_canonical_board()
    assert board.shape == (8, 8, 14)


def test_chess_game_canonical_board_has_float32_dtype(fresh_game: ChessGame):
    """Test that get_canonical_board() returns float32 dtype."""
    board = fresh_game.get_canonical_board()
    assert board.dtype == np.float32


def test_chess_game_canonical_board_flips_for_black_to_move():
    """Test that board is flipped when black to move (canonical form)."""
    # White to move
    game_white = ChessGame()
    board_white = game_white.get_canonical_board()

    # Black to move (after e4)
    game_black = ChessGame()
    game_black.make_move(chess.Move.from_uci("e2e4"))
    board_black = game_black.get_canonical_board()

    # Boards should be different (black's view is flipped)
    assert not np.array_equal(board_white, board_black)


def test_chess_game_canonical_board_piece_planes_encode_correctly():
    """Test that piece planes encode pieces correctly."""
    game = ChessGame()
    board = game.get_canonical_board()

    # Plane 0: Current player's pawns (white)
    # White pawns on rank 2 (index 6 in array, since rank 8 is index 0)
    assert np.sum(board[6, :, 0]) == 8  # 8 white pawns on rank 2

    # Plane 6: Opponent's pawns (black)
    # Black pawns on rank 7 (index 1)
    assert np.sum(board[1, :, 6]) == 8  # 8 black pawns on rank 7


def test_chess_game_canonical_board_all_piece_types_encoded_correctly():
    """Test that all piece types are encoded in correct planes for both colors."""
    # Standard starting position with all piece types in known locations
    # White: K on e1, Q on d1, R on a1/h1, B on c1/f1, N on b1/g1, pawns on rank 2
    # Black: K on e8, Q on d8, R on a8/h8, B on c8/f8, N on b8/g8, pawns on rank 7
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    game = ChessGame(fen=fen)
    board = game.get_canonical_board()

    # White to move, so white = current player (planes 0-5), black = opponent (planes 6-11)
    # Plane mapping: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King

    # Verify white pieces (current player, planes 0-5)
    # Pawns (plane 0): rank 2 = array index 6
    assert np.sum(board[6, :, 0]) == 8, "White should have 8 pawns on rank 2"

    # Knights (plane 1): b1 and g1
    # b1 = file 1, rank 1 = array index 7
    # g1 = file 6, rank 1 = array index 7
    assert board[7, 1, 1] == 1.0, "White knight should be on b1"
    assert board[7, 6, 1] == 1.0, "White knight should be on g1"
    assert np.sum(board[:, :, 1]) == 2, "White should have 2 knights"

    # Bishops (plane 2): c1 and f1
    # c1 = file 2, rank 1 = array index 7
    # f1 = file 5, rank 1 = array index 7
    assert board[7, 2, 2] == 1.0, "White bishop should be on c1"
    assert board[7, 5, 2] == 1.0, "White bishop should be on f1"
    assert np.sum(board[:, :, 2]) == 2, "White should have 2 bishops"

    # Rooks (plane 3): a1 and h1
    # a1 = file 0, rank 1 = array index 7
    # h1 = file 7, rank 1 = array index 7
    assert board[7, 0, 3] == 1.0, "White rook should be on a1"
    assert board[7, 7, 3] == 1.0, "White rook should be on h1"
    assert np.sum(board[:, :, 3]) == 2, "White should have 2 rooks"

    # Queen (plane 4): d1
    # d1 = file 3, rank 1 = array index 7
    assert board[7, 3, 4] == 1.0, "White queen should be on d1"
    assert np.sum(board[:, :, 4]) == 1, "White should have 1 queen"

    # King (plane 5): e1
    # e1 = file 4, rank 1 = array index 7
    assert board[7, 4, 5] == 1.0, "White king should be on e1"
    assert np.sum(board[:, :, 5]) == 1, "White should have 1 king"

    # Verify black pieces (opponent, planes 6-11)
    # Pawns (plane 6): rank 7 = array index 1
    assert np.sum(board[1, :, 6]) == 8, "Black should have 8 pawns on rank 7"

    # Knights (plane 7): b8 and g8
    # b8 = file 1, rank 8 = array index 0
    # g8 = file 6, rank 8 = array index 0
    assert board[0, 1, 7] == 1.0, "Black knight should be on b8"
    assert board[0, 6, 7] == 1.0, "Black knight should be on g8"
    assert np.sum(board[:, :, 7]) == 2, "Black should have 2 knights"

    # Bishops (plane 8): c8 and f8
    # c8 = file 2, rank 8 = array index 0
    # f8 = file 5, rank 8 = array index 0
    assert board[0, 2, 8] == 1.0, "Black bishop should be on c8"
    assert board[0, 5, 8] == 1.0, "Black bishop should be on f8"
    assert np.sum(board[:, :, 8]) == 2, "Black should have 2 bishops"

    # Rooks (plane 9): a8 and h8
    # a8 = file 0, rank 8 = array index 0
    # h8 = file 7, rank 8 = array index 0
    assert board[0, 0, 9] == 1.0, "Black rook should be on a8"
    assert board[0, 7, 9] == 1.0, "Black rook should be on h8"
    assert np.sum(board[:, :, 9]) == 2, "Black should have 2 rooks"

    # Queen (plane 10): d8
    # d8 = file 3, rank 8 = array index 0
    assert board[0, 3, 10] == 1.0, "Black queen should be on d8"
    assert np.sum(board[:, :, 10]) == 1, "Black should have 1 queen"

    # King (plane 11): e8
    # e8 = file 4, rank 8 = array index 0
    assert board[0, 4, 11] == 1.0, "Black king should be on e8"
    assert np.sum(board[:, :, 11]) == 1, "Black should have 1 king"


def test_chess_game_canonical_board_real_game_position_after_e4():
    """Test canonical board representation in a real game after e2-e4."""
    # Start with standard position, white plays e2-e4
    # This creates an en passant square and tests black's perspective
    game = ChessGame()
    game.make_move(chess.Move.from_uci("e2e4"))
    # Now black to move
    board = game.get_canonical_board()

    # Black to move, so black = current player (planes 0-5), white = opponent (planes 6-11)
    # Only ranks flip, files stay the same (e-file stays e-file)

    # 1) Current player's king (black king on e8) should be at board[7, 4, 5]
    assert board[7, 4, 5] == 1.0, "Black king should be at [7, 4, 5]"

    # 2) Opponent's king (white king on e1) should be at board[0, 4, 11]
    assert board[0, 4, 11] == 1.0, "White king should be at [0, 4, 11]"

    # 3) Current player's pawns (black, 8 pawns on rank 7) should be on array index 6
    assert np.sum(board[6, :, 0]) == 8, "Black should have 8 pawns on rank 6"

    # 4) Opponent's pawns (white, 7 on rank 2, 1 on e4)
    # White's rank 2 → flipped to array index 1
    # White's e4 pawn → flipped to array index 3, file 4
    assert np.sum(board[1, :, 6]) == 7, "White should have 7 pawns on rank 1 (flipped)"
    assert board[3, 4, 6] == 1.0, "White should have 1 pawn on e4 at [3, 4, 6]"
    assert np.sum(board[:, :, 6]) == 8, "White should have 8 pawns total"

    # 5) En passant square on e3 (file 4, rank 3)
    # e3: rank 3 → array index 5 → flipped: 7 - 5 = 2
    assert board[2, 4, 13] == 1.0, "En passant square should be at [2, 4, 13]"
    assert np.sum(board[:, :, 13]) == 1, "Should have exactly 1 en passant square"


def test_chess_game_canonical_board_repetition_plane_values():
    """Test that repetition plane (plane 12) correctly tracks position repetitions."""
    game = ChessGame()

    # Initial position: no repetition (first occurrence)
    board = game.get_canonical_board()
    assert np.all(board[:, :, 12] == 1.0 / 3.0), "Initial position should have repetition value 1/3"

    # Make moves that will allow us to repeat the position
    # Use knight moves to cycle back to the same position
    game.make_move(chess.Move.from_uci("g1f3"))  # White knight out
    game.make_move(chess.Move.from_uci("g8f6"))  # Black knight out
    game.make_move(chess.Move.from_uci("f3g1"))  # White knight back
    game.make_move(chess.Move.from_uci("f6g8"))  # Black knight back

    # Now we're back to the starting position (second occurrence)
    board = game.get_canonical_board()
    assert np.all(board[:, :, 12] == 2.0 / 3.0), "Second occurrence should have repetition value 2/3"

    # Repeat the same sequence again
    game.make_move(chess.Move.from_uci("g1f3"))
    game.make_move(chess.Move.from_uci("g8f6"))
    game.make_move(chess.Move.from_uci("f3g1"))
    game.make_move(chess.Move.from_uci("f6g8"))

    # Third occurrence (threefold repetition)
    board = game.get_canonical_board()
    assert np.all(board[:, :, 12] == 1.0), "Third occurrence should have repetition value 1.0"

    # Verify we can claim draw
    assert game.board.can_claim_draw(), "Should be able to claim draw after threefold repetition"


def test_chess_game_move_encoding_decoding_roundtrip(fresh_game: ChessGame):
    """Test that move encoding/decoding preserves the move."""
    legal_moves = fresh_game.get_legal_moves()

    for move in legal_moves:
        index = fresh_game.get_move_index(move)
        decoded_move = fresh_game.get_move_from_index(index)
        assert decoded_move == move


def test_chess_game_move_index_in_valid_range(fresh_game: ChessGame):
    """Test that all move indices are in range [0, 4671]."""
    legal_moves = fresh_game.get_legal_moves()

    for move in legal_moves:
        index = fresh_game.get_move_index(move)
        assert 0 <= index < 4672


def test_chess_game_all_legal_moves_have_unique_indices(fresh_game: ChessGame):
    """Test that all legal moves produce unique indices."""
    legal_moves = fresh_game.get_legal_moves()
    indices = [fresh_game.get_move_index(move) for move in legal_moves]

    # All indices should be unique
    assert len(indices) == len(set(indices))


def test_chess_game_illegal_move_index_raises_error(fresh_game: ChessGame):
    """Test that get_move_from_index() raises error for illegal move index."""
    # Find an index that's not a legal move
    legal_moves = fresh_game.get_legal_moves()
    legal_indices = {fresh_game.get_move_index(move) for move in legal_moves}

    # Find first illegal index
    illegal_index = None
    for i in range(4672):
        if i not in legal_indices:
            illegal_index = i
            break

    if illegal_index is not None:
        with pytest.raises(ValueError):
            fresh_game.get_move_from_index(illegal_index)


# =============================================================================
# G. SPECIAL MOVES TESTS
# =============================================================================

def test_chess_game_castling_kingside_legal(castling_test_position: ChessGame):
    """Test that kingside castling is legal when conditions met."""
    legal_moves = castling_test_position.get_legal_moves()
    kingside_castle = chess.Move.from_uci("e1g1")
    assert kingside_castle in legal_moves


def test_chess_game_castling_queenside_legal(castling_test_position: ChessGame):
    """Test that queenside castling is legal when conditions met."""
    legal_moves = castling_test_position.get_legal_moves()
    queenside_castle = chess.Move.from_uci("e1c1")
    assert queenside_castle in legal_moves


def test_chess_game_castling_rights_lost_after_king_moves(castling_test_position: ChessGame):
    """Test that castling rights are lost after king moves."""
    # Move king to f1
    castling_test_position.make_move(chess.Move.from_uci("e1f1"))
    # Move black king
    castling_test_position.make_move(chess.Move.from_uci("e8f8"))
    # Move white king back to e1
    castling_test_position.make_move(chess.Move.from_uci("f1e1"))

    # Castling should no longer be legal
    legal_moves = castling_test_position.get_legal_moves()
    kingside_castle = chess.Move.from_uci("e1g1")
    queenside_castle = chess.Move.from_uci("e1c1")

    assert kingside_castle not in legal_moves
    assert queenside_castle not in legal_moves


def test_chess_game_castling_rights_lost_after_rook_moves():
    """Test that castling rights lost after rook moves."""
    game = ChessGame(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    # Move h1 rook
    game.make_move(chess.Move.from_uci("h1g1"))
    game.make_move(chess.Move.from_uci("a8b8"))  # Black move
    game.make_move(chess.Move.from_uci("g1h1"))  # Move rook back

    # Kingside castling should be illegal now
    legal_moves = game.get_legal_moves()
    kingside_castle = chess.Move.from_uci("e1g1")
    assert kingside_castle not in legal_moves


def test_chess_game_cannot_castle_through_check():
    """Test that castling through check is illegal."""
    # Position where castling would move king through attacked square
    fen = "3rkr2/8/8/8/8/8/8/R3K2R w KQ - 0 1"
    game = ChessGame(fen=fen)

    # Queenside and kingside castle would move king through attacked square
    legal_moves = game.get_legal_moves()
    queenside_castle = chess.Move.from_uci("e1c1")
    kingside_castle = chess.Move.from_uci("e1g1")
    assert queenside_castle not in legal_moves
    assert kingside_castle not in legal_moves


def test_chess_game_cannot_castle_out_of_check():
    """Test that castling out of check is illegal."""
    # King is in check
    fen = "4k3/4r3/8/8/8/8/8/R3K2R w KQ - 0 1"
    game = ChessGame(fen=fen)

    # King is in check from e7 rook, cannot castle
    legal_moves = game.get_legal_moves()
    kingside_castle = chess.Move.from_uci("e1g1")
    queenside_castle = chess.Move.from_uci("e1c1")
    assert kingside_castle not in legal_moves
    assert queenside_castle not in legal_moves


def test_chess_game_cannot_castle_into_check():
    """Test that castling into check is illegal."""
    # Castling would put king in check
    fen = "4k3/2r5/6r1/8/8/8/8/R3K2R w KQ - 0 1"
    game = ChessGame(fen=fen)

    legal_moves = game.get_legal_moves()
    kingside_castle = chess.Move.from_uci("e1g1")
    queenside_castle = chess.Move.from_uci("e1c1")
    assert kingside_castle not in legal_moves
    assert queenside_castle not in legal_moves


def test_chess_game_en_passant_capture_legal(en_passant_position: ChessGame):
    """Test that en passant capture is legal when available."""
    legal_moves = en_passant_position.get_legal_moves()
    en_passant_move = chess.Move.from_uci("e5d6")
    assert en_passant_move in legal_moves


def test_chess_game_en_passant_opportunity_expires():
    """Test that en passant opportunity expires after one move."""
    game = ChessGame()

    # Set up en passant opportunity
    game.make_move(chess.Move.from_uci("e2e4"))
    game.make_move(chess.Move.from_uci("a7a6"))  # Black makes different move
    game.make_move(chess.Move.from_uci("e4e5"))
    game.make_move(chess.Move.from_uci("d7d5"))  # Black pawn double move

    # En passant should be available
    legal_moves = game.get_legal_moves()
    en_passant = chess.Move.from_uci("e5d6")
    assert en_passant in legal_moves

    # Make a different move
    game.make_move(chess.Move.from_uci("g1f3"))
    game.make_move(chess.Move.from_uci("a6a5"))

    # En passant should no longer be available
    legal_moves = game.get_legal_moves()
    # Can't capture en passant anymore (different position now)
    assert en_passant not in legal_moves


def test_chess_game_pawn_promotion_to_queen(promotion_position: ChessGame):
    """Test pawn promotion to queen."""
    promotion_move = chess.Move.from_uci("a7a8q")
    legal_moves = promotion_position.get_legal_moves()
    assert promotion_move in legal_moves

    promotion_position.make_move(promotion_move)
    # Verify queen is on a8
    state = promotion_position.get_state()
    assert "Q" in state


def test_chess_game_pawn_promotion_to_rook(promotion_position: ChessGame):
    """Test pawn promotion to rook."""
    promotion_move = chess.Move.from_uci("a7a8r")
    legal_moves = promotion_position.get_legal_moves()
    assert promotion_move in legal_moves


def test_chess_game_pawn_promotion_to_bishop(promotion_position: ChessGame):
    """Test pawn promotion to bishop."""
    promotion_move = chess.Move.from_uci("a7a8b")
    legal_moves = promotion_position.get_legal_moves()
    assert promotion_move in legal_moves


def test_chess_game_pawn_promotion_to_knight(promotion_position: ChessGame):
    """Test pawn promotion to knight (underpromotion)."""
    promotion_move = chess.Move.from_uci("a7a8n")
    legal_moves = promotion_position.get_legal_moves()
    assert promotion_move in legal_moves


def test_chess_game_promotion_encoding_distinguishes_piece_types(promotion_position: ChessGame):
    """Test that move encoding distinguishes promotion types."""
    queen_promo = chess.Move.from_uci("a7a8q")
    rook_promo = chess.Move.from_uci("a7a8r")
    bishop_promo = chess.Move.from_uci("a7a8b")
    knight_promo = chess.Move.from_uci("a7a8n")

    indices = [
        promotion_position.get_move_index(queen_promo),
        promotion_position.get_move_index(rook_promo),
        promotion_position.get_move_index(bishop_promo),
        promotion_position.get_move_index(knight_promo),
    ]

    # All indices should be different
    assert len(indices) == len(set(indices))


# =============================================================================
# H. EDGE CASE TESTS
# =============================================================================

def test_chess_game_king_returns_to_e1_after_castling_no_longer_can_castle():
    """Test edge case: king moves back to e1 after castling rights lost."""
    # Position with pawns on ranks 2 and 7 to prevent rook interference
    game = ChessGame(fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

    # Execute move sequence to castle and return king to e1
    game.make_move(chess.Move.from_uci("e1g1"))  # White castles kingside
    game.make_move(chess.Move.from_uci("e8g8"))  # Black castles kingside
    game.make_move(chess.Move.from_uci("f2f3"))  # Open path for white king
    game.make_move(chess.Move.from_uci("f7f6"))  # Open path for black king
    game.make_move(chess.Move.from_uci("g1f2"))  # White king to f2
    game.make_move(chess.Move.from_uci("g8f7"))  # Black king to f7
    game.make_move(chess.Move.from_uci("f1h1"))  # White rook to h1
    game.make_move(chess.Move.from_uci("f8h8"))  # Black rook to h8
    game.make_move(chess.Move.from_uci("f2e1"))  # White king returns to e1
    game.make_move(chess.Move.from_uci("f7e8"))  # Black king returns to e8

    # King is on e1 but castling should not be legal (rights lost after castling)
    legal_moves = game.get_legal_moves()
    assert chess.Move.from_uci("e1g1") not in legal_moves
    assert chess.Move.from_uci("e1c1") not in legal_moves


def test_chess_game_threefold_repetition_draw():
    """Test that threefold repetition results in draw."""
    game = ChessGame()

    # Repeat position 3 times with knight moves
    for _ in range(2):
        game.make_move(chess.Move.from_uci("g1f3"))
        game.make_move(chess.Move.from_uci("g8f6"))
        game.make_move(chess.Move.from_uci("f3g1"))
        game.make_move(chess.Move.from_uci("f6g8"))

    # After third repetition, can claim draw
    # Note: python-chess doesn't automatically end game on repetition
    # but is_draw() should detect it
    game.make_move(chess.Move.from_uci("g1f3"))
    game.make_move(chess.Move.from_uci("g8f6"))
    game.make_move(chess.Move.from_uci("f3g1"))
    game.make_move(chess.Move.from_uci("f6g8"))

    # Check if threefold repetition can be claimed
    assert game.board.can_claim_draw()


def test_chess_game_fifty_move_rule():
    """Test that fifty-move rule results in draw."""
    # Start from endgame position
    game = ChessGame(fen="7k/8/8/8/8/8/4Q3/4K3 w - - 0 1")

    # Make 50 moves without pawn move or capture
    # This is tedious to test fully, but we can verify the concept
    # by checking FEN halfmove clock
    for _ in range(25):
        game.make_move(chess.Move.from_uci("e2e3"))
        game.make_move(chess.Move.from_uci("h8g8"))
        game.make_move(chess.Move.from_uci("e3e2"))
        game.make_move(chess.Move.from_uci("g8h8"))

    # Should be draw by fifty-move rule
    assert game.board.halfmove_clock >= 50
    assert game.is_draw() or game.board.can_claim_draw()


def test_chess_game_discovered_check_position():
    """Test that discovered check is handled correctly."""
    # Position with potential discovered check
    fen = "3k4/8/8/8/8/8/3B4/3RK3 w - - 0 1"
    game = ChessGame(fen=fen)

    # Moving the bishop on d2 creates a legal, discovered check
    legal_moves = game.get_legal_moves()

    assert chess.Move.from_uci("d2f4") in legal_moves
    assert chess.Move.from_uci("d2g5") in legal_moves


def test_chess_game_pinned_piece_cannot_move():
    """Test that pinned pieces have restricted movement."""
    # Position with pinned white knight
    fen = "4k3/8/8/8/4r3/8/4N3/4K3 w - - 0 1"
    game = ChessGame(fen=fen)

    # Knight on e2 is pinned by rook on e4
    legal_moves = game.get_legal_moves()
    assert chess.Move.from_uci("e2g1") not in legal_moves
    assert chess.Move.from_uci("e2g3") not in legal_moves
    assert chess.Move.from_uci("e2f4") not in legal_moves
    assert chess.Move.from_uci("e2d4") not in legal_moves
    assert chess.Move.from_uci("e2c3") not in legal_moves
    assert chess.Move.from_uci("e2c1") not in legal_moves


def test_chess_game_complex_endgame_position():
    """Test complex endgame with multiple piece types."""
    fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    game = ChessGame(fen=fen)

    legal_moves = game.get_legal_moves()

    # Verify game state is handled correctly
    assert len(legal_moves) > 0
    assert not game.is_game_over()


def test_chess_game_handles_unicode_piece_symbols():
    """Test that game handles unicode piece symbols in FEN correctly."""
    game = ChessGame()

    # Get state and verify it doesn't contain unicode symbols
    state = game.get_state()
    assert isinstance(state, str)
    # FEN should use ASCII characters only
    assert all(ord(c) < 128 for c in state)
