"""
Comprehensive tests for ChessGame component.

Test classes:
- TestChessGameInitialization
- TestChessGameMoveExecution
- TestChessGameLegalMoves
- TestChessGameStatus
- TestChessGameCloning
- TestChessGameNeuralNetworkInterface
- TestChessGameSpecialMoves
- TestChessGameEdgeCases
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
def endgame_position() -> ChessGame:
    """Simple K+Q vs K endgame."""
    return ChessGame(fen="7k/8/8/8/8/8/4Q3/4K3 w - - 0 1")


@pytest.fixture
def stalemate_position() -> ChessGame:
    """Known stalemate position."""
    return ChessGame(fen="7k/5Q2/5K2/8/8/8/8/8 b - - 0 1")


@pytest.fixture
def checkmate_position() -> ChessGame:
    """Known checkmate position (Fool's mate - white checkmated)."""
    return ChessGame(fen="rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")


@pytest.fixture
def castling_position() -> ChessGame:
    """Position where both sides can castle both ways."""
    return ChessGame(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")


@pytest.fixture
def en_passant_position() -> ChessGame:
    """Position ready for en passant capture."""
    return ChessGame(fen="rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1")


@pytest.fixture
def promotion_position() -> ChessGame:
    """White pawn ready to promote."""
    return ChessGame(fen="8/P7/8/8/8/8/8/K6k w - - 0 1")


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

    def test_valid_none_fen_creates_starting_position(self) -> None:
        """ChessGame(fen=None) creates starting position."""
        game = ChessGame(fen=None)
        expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert game.get_state() == expected_fen

    @pytest.mark.parametrize('fen', [
        pytest.param(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            id="italian_game"
        ),
        pytest.param(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            id="kings_pawn_opening"
        ),
        pytest.param(
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            id="king_pawn_endgame"
        ),
        pytest.param(
            "rnbqkbnr/pppp1ppp/8/8/2PpP3/8/PP3PPP/RNBQKBNR b KQkq c3 0 1",
            id="en_passant_available"
        ),
    ])
    def test_valid_custom_fen_accepted(self, fen: str) -> None:
        """Custom FEN strings are parsed correctly."""
        game = ChessGame(fen=fen)
        assert game.get_state() == fen

    def test_valid_fen_roundtrip_preserves_state(self) -> None:
        """Exporting and re-importing FEN preserves game state."""
        original_fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        game1 = ChessGame(fen=original_fen)
        exported_fen = game1.get_state()
        game2 = ChessGame(fen=exported_fen)
        assert game2.get_state() == original_fen


class TestChessGameMoveExecution:
    """Tests for move execution via make_move() and undo_move()."""

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
        assert starting_position.board.piece_at(square).symbol() == piece_symbol

    def test_valid_move_changes_side_to_move(self, starting_position: ChessGame) -> None:
        """Making a move switches the side to move."""
        assert starting_position.board.turn == chess.WHITE
        starting_position.make_move(chess.Move.from_uci("e2e4"))
        assert starting_position.board.turn == chess.BLACK

    def test_valid_undo_restores_previous_state(self, starting_position: ChessGame) -> None:
        """undo_move() restores the state before the last move."""
        initial_fen = starting_position.get_state()
        starting_position.make_move(chess.Move.from_uci("e2e4"))
        starting_position.undo_move()
        assert starting_position.get_state() == initial_fen

    def test_valid_multiple_moves_and_undos(self, starting_position: ChessGame) -> None:
        """Multiple moves and undos work correctly."""
        state0 = starting_position.get_state()

        starting_position.make_move(chess.Move.from_uci("e2e4"))
        state1 = starting_position.get_state()

        starting_position.make_move(chess.Move.from_uci("e7e5"))
        state2 = starting_position.get_state()

        starting_position.make_move(chess.Move.from_uci("g1f3"))

        starting_position.undo_move()
        assert starting_position.get_state() == state2

        starting_position.undo_move()
        assert starting_position.get_state() == state1

        starting_position.undo_move()
        assert starting_position.get_state() == state0

    def test_valid_move_history_tracking(self, starting_position: ChessGame) -> None:
        """Move history is tracked correctly."""
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("e7e5"),
            chess.Move.from_uci("g1f3"),
        ]

        for move in moves:
            starting_position.make_move(move)

        # Undo all moves
        for _ in moves:
            starting_position.undo_move()

        expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert starting_position.get_state() == expected_fen

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

    # -------------------------------------------------------------------------
    # Valid behavior tests
    # -------------------------------------------------------------------------

    def test_valid_starting_position_has_twenty_moves(
        self,
        starting_position: ChessGame
    ) -> None:
        """Starting position has exactly 20 legal moves."""
        legal_moves = starting_position.get_legal_moves()
        assert len(legal_moves) == 20

    def test_valid_returns_list_of_chess_moves(
        self,
        starting_position: ChessGame
    ) -> None:
        """get_legal_moves() returns list[chess.Move]."""
        legal_moves = starting_position.get_legal_moves()
        assert isinstance(legal_moves, list)
        assert all(isinstance(move, chess.Move) for move in legal_moves)

    def test_valid_mask_has_correct_shape(self, starting_position: ChessGame) -> None:
        """legal_moves_mask has shape (4672,)."""
        mask = starting_position.get_legal_moves_mask()
        assert mask.shape == (4672,)
        assert mask.dtype == bool or mask.dtype == np.bool_

    def test_valid_mask_matches_legal_moves(self, starting_position: ChessGame) -> None:
        """legal_moves_mask True values correspond to actual legal moves."""
        legal_moves = starting_position.get_legal_moves()
        mask = starting_position.get_legal_moves_mask()

        assert np.sum(mask) == len(legal_moves)

        for move in legal_moves:
            move_index = starting_position.get_move_index(move)
            assert mask[move_index], f"Move {move.uci()} should be True in mask"

    @pytest.mark.parametrize('fen,expected_count', [
        pytest.param(
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            33,
            id="italian_game_white"
        ),
        pytest.param(
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            7,
            id="simple_endgame"
        ),
    ])
    def test_valid_position_move_count(self, fen: str, expected_count: int) -> None:
        """Positions have correct number of legal moves."""
        game = ChessGame(fen=fen)
        assert len(game.get_legal_moves()) == expected_count

    def test_valid_move_count_consistency(self, endgame_position: ChessGame) -> None:
        """Legal move count is consistent across different methods."""
        legal_moves = endgame_position.get_legal_moves()
        mask = endgame_position.get_legal_moves_mask()
        assert len(legal_moves) == np.sum(mask)

    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------

    def test_edge_checkmate_has_zero_moves(self, checkmate_position: ChessGame) -> None:
        """Checkmate position has zero legal moves."""
        legal_moves = checkmate_position.get_legal_moves()
        assert len(legal_moves) == 0

    def test_edge_stalemate_has_zero_moves(self, stalemate_position: ChessGame) -> None:
        """Stalemate position has zero legal moves."""
        legal_moves = stalemate_position.get_legal_moves()
        assert len(legal_moves) == 0


class TestChessGameStatus:
    """Tests for game status detection (checkmate, stalemate, draws)."""

    # -------------------------------------------------------------------------
    # Valid behavior tests
    # -------------------------------------------------------------------------

    def test_valid_starting_position_not_game_over(
        self,
        starting_position: ChessGame
    ) -> None:
        """Starting position is not game over."""
        assert not starting_position.is_game_over()

    def test_valid_checkmate_detected(self, checkmate_position: ChessGame) -> None:
        """Checkmate is correctly detected."""
        assert checkmate_position.is_game_over()

    def test_valid_stalemate_detected(self, stalemate_position: ChessGame) -> None:
        """Stalemate is correctly detected as game over and draw."""
        assert stalemate_position.is_game_over()
        assert stalemate_position.is_draw()

    def test_valid_result_none_when_not_over(self, starting_position: ChessGame) -> None:
        """get_result() returns None when game is not over."""
        assert starting_position.get_result() is None

    def test_valid_result_checkmate_loser_perspective(self) -> None:
        """get_result() returns -1.0 for checkmated player."""
        # Fool's mate: white is checkmated, it's white's turn
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        game = ChessGame(fen=fen)
        result = game.get_result()
        assert result == -1.0

    def test_valid_result_stalemate_returns_zero(
        self,
        stalemate_position: ChessGame
    ) -> None:
        """get_result() returns 0.0 for stalemate."""
        result = stalemate_position.get_result()
        assert result == 0.0

    # -------------------------------------------------------------------------
    # Insufficient material tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize('fen', [
        pytest.param("7k/8/8/8/8/8/8/K7 w - - 0 1", id="king_vs_king"),
        pytest.param("7k/8/8/8/8/8/4N3/K7 w - - 0 1", id="king_knight_vs_king"),
        pytest.param("7k/8/8/8/8/8/4B3/K7 w - - 0 1", id="king_bishop_vs_king"),
    ])
    def test_valid_insufficient_material_draw(self, fen: str) -> None:
        """Insufficient material positions are detected as draws."""
        game = ChessGame(fen=fen)
        assert game.is_draw()

    # -------------------------------------------------------------------------
    # Famous checkmate sequence tests
    # -------------------------------------------------------------------------

    def test_valid_fools_mate_sequence(self) -> None:
        """Fool's Mate sequence: f3, e6, g4, Qh4# - fastest checkmate."""
        game = ChessGame()

        game.make_move(chess.Move.from_uci("f2f3"))
        assert not game.is_game_over()

        game.make_move(chess.Move.from_uci("e7e6"))
        assert not game.is_game_over()

        game.make_move(chess.Move.from_uci("g2g4"))
        assert not game.is_game_over()

        game.make_move(chess.Move.from_uci("d8h4"))
        assert game.is_game_over()
        assert len(game.get_legal_moves()) == 0
        assert game.get_result() == -1.0

    def test_valid_scholars_mate_sequence(self) -> None:
        """Scholar's Mate sequence: e4, e5, Bc4, Nc6, Qh5, Nf6, Qxf7#."""
        game = ChessGame()
        moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]

        for move_uci in moves[:-1]:
            game.make_move(chess.Move.from_uci(move_uci))
            assert not game.is_game_over()

        game.make_move(chess.Move.from_uci(moves[-1]))
        assert game.is_game_over()
        assert game.get_result() == -1.0


class TestChessGameCloning:
    """Tests for game state cloning."""

    # -------------------------------------------------------------------------
    # Valid behavior tests
    # -------------------------------------------------------------------------

    def test_valid_clone_creates_independent_copy(
        self,
        starting_position: ChessGame
    ) -> None:
        """clone() creates an independent copy."""
        clone = starting_position.clone()
        starting_position.make_move(chess.Move.from_uci("e2e4"))

        expected_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        assert clone.get_state() == expected_fen

    def test_valid_clone_preserves_all_state(self, castling_position: ChessGame) -> None:
        """Clone preserves position, turn, castling rights, en passant."""
        original_fen = castling_position.get_state()
        clone = castling_position.clone()
        assert clone.get_state() == original_fen

    def test_valid_modifying_clone_does_not_affect_original(
        self,
        starting_position: ChessGame
    ) -> None:
        """Modifying clone doesn't affect original game."""
        original_state = starting_position.get_state()
        clone = starting_position.clone()

        clone.make_move(chess.Move.from_uci("e2e4"))
        clone.make_move(chess.Move.from_uci("e7e5"))
        clone.make_move(chess.Move.from_uci("g1f3"))

        assert starting_position.get_state() == original_state

    def test_valid_clone_of_clone_works(self, starting_position: ChessGame) -> None:
        """Cloning a clone works correctly."""
        clone1 = starting_position.clone()
        clone1.make_move(chess.Move.from_uci("e2e4"))

        clone2 = clone1.clone()
        clone2.make_move(chess.Move.from_uci("e7e5"))

        # All three should have different states
        assert starting_position.get_state() != clone1.get_state()
        assert clone1.get_state() != clone2.get_state()
        assert starting_position.get_state() != clone2.get_state()


class TestChessGameNeuralNetworkInterface:
    """Tests for neural network input/output interface."""

    # -------------------------------------------------------------------------
    # Valid behavior tests
    # -------------------------------------------------------------------------

    def test_valid_canonical_board_shape(self, starting_position: ChessGame) -> None:
        """Canonical board has correct shape (8, 8, 18)."""
        board = starting_position.get_canonical_board()
        assert board.shape == (8, 8, 18)

    def test_valid_canonical_board_dtype(self, starting_position: ChessGame) -> None:
        """Canonical board has float32 dtype."""
        board = starting_position.get_canonical_board()
        assert board.dtype == np.float32

    def test_valid_move_encoding_roundtrip(self, starting_position: ChessGame) -> None:
        """Move encoding/decoding preserves the move."""
        legal_moves = starting_position.get_legal_moves()

        for move in legal_moves:
            index = starting_position.get_move_index(move)
            decoded_move = starting_position.get_move_from_index(index)
            assert decoded_move == move

    def test_valid_move_index_in_range(self, starting_position: ChessGame) -> None:
        """All move indices are in range [0, 4671]."""
        legal_moves = starting_position.get_legal_moves()

        for move in legal_moves:
            index = starting_position.get_move_index(move)
            assert 0 <= index < 4672

    def test_valid_all_moves_have_unique_indices(
        self,
        starting_position: ChessGame
    ) -> None:
        """All legal moves produce unique indices."""
        legal_moves = starting_position.get_legal_moves()
        indices = [starting_position.get_move_index(move) for move in legal_moves]
        assert len(indices) == len(set(indices))

    # -------------------------------------------------------------------------
    # Canonical board representation tests
    # -------------------------------------------------------------------------

    def test_canonical_board_flips_for_black(self) -> None:
        """Board is flipped when black to move (canonical representation)."""
        game_white = ChessGame()
        board_white = game_white.get_canonical_board()

        game_black = ChessGame()
        game_black.make_move(chess.Move.from_uci("e2e4"))
        board_black = game_black.get_canonical_board()

        assert not np.array_equal(board_white, board_black)

    def test_canonical_piece_planes_encode_correctly(self) -> None:
        """Piece planes encode pieces correctly."""
        game = ChessGame()
        board = game.get_canonical_board()

        # Plane 0: Current player's pawns (white) on rank 2 (array index 6)
        assert np.sum(board[6, :, 0]) == 8

        # Plane 6: Opponent's pawns (black) on rank 7 (array index 1)
        assert np.sum(board[1, :, 6]) == 8

    def test_canonical_all_piece_types_encoded(self) -> None:
        """All piece types are encoded in correct planes."""
        game = ChessGame()
        board = game.get_canonical_board()

        # White pieces (planes 0-5): P, N, B, R, Q, K
        assert np.sum(board[6, :, 0]) == 8  # 8 pawns
        assert np.sum(board[:, :, 1]) == 2  # 2 knights
        assert np.sum(board[:, :, 2]) == 2  # 2 bishops
        assert np.sum(board[:, :, 3]) == 2  # 2 rooks
        assert np.sum(board[:, :, 4]) == 1  # 1 queen
        assert np.sum(board[:, :, 5]) == 1  # 1 king

        # Black pieces (planes 6-11)
        assert np.sum(board[1, :, 6]) == 8  # 8 pawns
        assert np.sum(board[:, :, 7]) == 2  # 2 knights
        assert np.sum(board[:, :, 8]) == 2  # 2 bishops
        assert np.sum(board[:, :, 9]) == 2  # 2 rooks
        assert np.sum(board[:, :, 10]) == 1  # 1 queen
        assert np.sum(board[:, :, 11]) == 1  # 1 king

    def test_canonical_after_e4_black_perspective(self) -> None:
        """Canonical board after e2-e4 shows black's perspective."""
        game = ChessGame()
        game.make_move(chess.Move.from_uci("e2e4"))
        board = game.get_canonical_board()

        # Black to move: black = current player (planes 0-5)
        # Black king on e8 should be at [7, 4, 5] (flipped)
        assert board[7, 4, 5] == 1.0

        # White king on e1 should be at [0, 4, 11] (opponent, flipped)
        assert board[0, 4, 11] == 1.0

        # En passant square on e3 should be marked in plane 13
        assert board[2, 4, 13] == 1.0
        assert np.sum(board[:, :, 13]) == 1

    def test_canonical_repetition_plane_values(self) -> None:
        """Repetition plane (plane 12) correctly tracks position repetitions."""
        game = ChessGame()

        # Initial position: first occurrence (1/3)
        board = game.get_canonical_board()
        assert np.all(board[:, :, 12] == 1.0 / 3.0)

        # Return to starting position
        game.make_move(chess.Move.from_uci("g1f3"))
        game.make_move(chess.Move.from_uci("g8f6"))
        game.make_move(chess.Move.from_uci("f3g1"))
        game.make_move(chess.Move.from_uci("f6g8"))

        # Second occurrence (2/3)
        board = game.get_canonical_board()
        assert np.all(board[:, :, 12] == 2.0 / 3.0)

        # Third occurrence (1.0)
        game.make_move(chess.Move.from_uci("g1f3"))
        game.make_move(chess.Move.from_uci("g8f6"))
        game.make_move(chess.Move.from_uci("f3g1"))
        game.make_move(chess.Move.from_uci("f6g8"))

        board = game.get_canonical_board()
        assert np.all(board[:, :, 12] == 1.0)
        assert game.board.can_claim_draw()

    # -------------------------------------------------------------------------
    # Castling rights plane tests
    # -------------------------------------------------------------------------

    def test_canonical_castling_all_rights_available(self) -> None:
        """Starting position has all castling rights (planes 14-17 all 1.0)."""
        game = ChessGame()
        board = game.get_canonical_board()

        # White to move: current player = white, opponent = black
        # Plane 14: current player kingside (white K)
        # Plane 15: current player queenside (white Q)
        # Plane 16: opponent kingside (black k)
        # Plane 17: opponent queenside (black q)
        assert np.all(board[:, :, 14] == 1.0), "Current player kingside should be available"
        assert np.all(board[:, :, 15] == 1.0), "Current player queenside should be available"
        assert np.all(board[:, :, 16] == 1.0), "Opponent kingside should be available"
        assert np.all(board[:, :, 17] == 1.0), "Opponent queenside should be available"

    def test_canonical_castling_no_rights(self) -> None:
        """Position with no castling rights has all castling planes as 0.0."""
        # Position with no castling rights
        game = ChessGame(fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1")
        board = game.get_canonical_board()

        assert np.all(board[:, :, 14] == 0.0), "No kingside castling for current player"
        assert np.all(board[:, :, 15] == 0.0), "No queenside castling for current player"
        assert np.all(board[:, :, 16] == 0.0), "No kingside castling for opponent"
        assert np.all(board[:, :, 17] == 0.0), "No queenside castling for opponent"

    def test_canonical_castling_partial_rights(self) -> None:
        """Position with partial castling rights encodes correctly."""
        # Only white kingside and black queenside available
        game = ChessGame(fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1")
        board = game.get_canonical_board()

        # White to move: current = white, opponent = black
        assert np.all(board[:, :, 14] == 1.0), "White kingside (K) available"
        assert np.all(board[:, :, 15] == 0.0), "White queenside (Q) not available"
        assert np.all(board[:, :, 16] == 0.0), "Black kingside (k) not available"
        assert np.all(board[:, :, 17] == 1.0), "Black queenside (q) available"

    def test_canonical_castling_perspective_flip_black_to_move(self) -> None:
        """Black to move: castling planes swap current/opponent perspective."""
        # All castling rights, black to move
        game = ChessGame(fen="r3k2r/pppppppp/8/8/4P3/8/PPPP1PPP/R3K2R b KQkq - 0 1")
        board = game.get_canonical_board()

        # Black to move: current = black, opponent = white
        # Plane 14: current player kingside = black k
        # Plane 15: current player queenside = black q
        # Plane 16: opponent kingside = white K
        # Plane 17: opponent queenside = white Q
        assert np.all(board[:, :, 14] == 1.0), "Black kingside (current) available"
        assert np.all(board[:, :, 15] == 1.0), "Black queenside (current) available"
        assert np.all(board[:, :, 16] == 1.0), "White kingside (opponent) available"
        assert np.all(board[:, :, 17] == 1.0), "White queenside (opponent) available"

    def test_canonical_castling_perspective_flip_partial_rights(self) -> None:
        """Black to move with partial rights: verify perspective swap."""
        # White has K only, black has q only, black to move
        game = ChessGame(fen="r3k2r/pppppppp/8/8/4P3/8/PPPP1PPP/R3K2R b Kq - 0 1")
        board = game.get_canonical_board()

        # Black to move: current = black, opponent = white
        assert np.all(board[:, :, 14] == 0.0), "Black kingside (k) not available"
        assert np.all(board[:, :, 15] == 1.0), "Black queenside (q) available"
        assert np.all(board[:, :, 16] == 1.0), "White kingside (K) available"
        assert np.all(board[:, :, 17] == 0.0), "White queenside (Q) not available"

    def test_canonical_castling_rights_lost_after_move(self, castling_position: ChessGame) -> None:
        """Castling rights update correctly after king moves."""
        # Initial state - all rights available
        board_before = castling_position.get_canonical_board()
        assert np.all(board_before[:, :, 14] == 1.0)
        assert np.all(board_before[:, :, 15] == 1.0)

        # Move white king - loses both white castling rights
        castling_position.make_move(chess.Move.from_uci("e1f1"))
        board_after = castling_position.get_canonical_board()

        # Now black to move: current = black, opponent = white
        # White (opponent) should have lost both castling rights
        assert np.all(board_after[:, :, 16] == 0.0), "White kingside lost"
        assert np.all(board_after[:, :, 17] == 0.0), "White queenside lost"
        # Black (current) should still have both
        assert np.all(board_after[:, :, 14] == 1.0), "Black kingside still available"
        assert np.all(board_after[:, :, 15] == 1.0), "Black queenside still available"

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_error_illegal_move_index(self, starting_position: ChessGame) -> None:
        """get_move_from_index() raises error for illegal move index."""
        legal_moves = starting_position.get_legal_moves()
        legal_indices = {starting_position.get_move_index(move) for move in legal_moves}

        for i in range(4672):
            if i not in legal_indices:
                with pytest.raises(ValueError):
                    starting_position.get_move_from_index(i)
                break


class TestChessGameSpecialMoves:
    """Tests for castling, en passant, and promotion."""

    # -------------------------------------------------------------------------
    # Castling tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize('castle_uci,rook_dest', [
        pytest.param("e1g1", chess.F1, id="white_kingside"),
        pytest.param("e1c1", chess.D1, id="white_queenside"),
    ])
    def test_valid_castling_legal(
        self,
        castling_position: ChessGame,
        castle_uci: str,
        rook_dest: int
    ) -> None:
        """Castling is legal when conditions met."""
        legal_moves = castling_position.get_legal_moves()
        castle_move = chess.Move.from_uci(castle_uci)
        assert castle_move in legal_moves

    def test_valid_castling_moves_rook(self, castling_position: ChessGame) -> None:
        """Castling moves both king and rook."""
        castling_position.make_move(chess.Move.from_uci("e1g1"))
        assert castling_position.board.piece_at(chess.F1).piece_type == chess.ROOK

    def test_valid_castling_rights_lost_after_king_moves(
        self,
        castling_position: ChessGame
    ) -> None:
        """Castling rights are lost after king moves."""
        castling_position.make_move(chess.Move.from_uci("e1f1"))
        castling_position.make_move(chess.Move.from_uci("e8f8"))
        castling_position.make_move(chess.Move.from_uci("f1e1"))

        legal_moves = castling_position.get_legal_moves()
        assert chess.Move.from_uci("e1g1") not in legal_moves
        assert chess.Move.from_uci("e1c1") not in legal_moves

    def test_valid_castling_rights_lost_after_rook_moves(self) -> None:
        """Castling rights lost after rook moves."""
        game = ChessGame(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

        game.make_move(chess.Move.from_uci("h1g1"))
        game.make_move(chess.Move.from_uci("a8b8"))
        game.make_move(chess.Move.from_uci("g1h1"))

        legal_moves = game.get_legal_moves()
        assert chess.Move.from_uci("e1g1") not in legal_moves

    @pytest.mark.parametrize('fen', [
        pytest.param(
            "3rkr2/8/8/8/8/8/8/R3K2R w KQ - 0 1",
            id="through_check"
        ),
        pytest.param(
            "4k3/4r3/8/8/8/8/8/R3K2R w KQ - 0 1",
            id="out_of_check"
        ),
        pytest.param(
            "4k3/2r5/6r1/8/8/8/8/R3K2R w KQ - 0 1",
            id="into_check"
        ),
    ])
    def test_edge_cannot_castle_illegally(self, fen: str) -> None:
        """Cannot castle through, out of, or into check."""
        game = ChessGame(fen=fen)
        legal_moves = game.get_legal_moves()
        assert chess.Move.from_uci("e1g1") not in legal_moves
        assert chess.Move.from_uci("e1c1") not in legal_moves

    # -------------------------------------------------------------------------
    # En passant tests
    # -------------------------------------------------------------------------

    def test_valid_en_passant_legal(self, en_passant_position: ChessGame) -> None:
        """En passant capture is legal when available."""
        legal_moves = en_passant_position.get_legal_moves()
        en_passant_move = chess.Move.from_uci("e5d6")
        assert en_passant_move in legal_moves

    def test_valid_en_passant_captures_pawn(self) -> None:
        """En passant capture removes the captured pawn."""
        game = ChessGame(fen="rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        game.make_move(chess.Move.from_uci("e5f6"))

        assert game.board.piece_at(chess.F5) is None
        assert game.board.piece_at(chess.F6).symbol() == "P"

    def test_edge_en_passant_expires(self) -> None:
        """En passant opportunity expires after one move."""
        game = ChessGame()

        game.make_move(chess.Move.from_uci("e2e4"))
        game.make_move(chess.Move.from_uci("a7a6"))
        game.make_move(chess.Move.from_uci("e4e5"))
        game.make_move(chess.Move.from_uci("d7d5"))

        # En passant available
        assert chess.Move.from_uci("e5d6") in game.get_legal_moves()

        # Make different move
        game.make_move(chess.Move.from_uci("g1f3"))
        game.make_move(chess.Move.from_uci("a6a5"))

        # En passant no longer available
        assert chess.Move.from_uci("e5d6") not in game.get_legal_moves()

    # -------------------------------------------------------------------------
    # Promotion tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize('promo_uci,promo_piece', [
        pytest.param("a7a8q", chess.QUEEN, id="queen"),
        pytest.param("a7a8r", chess.ROOK, id="rook"),
        pytest.param("a7a8b", chess.BISHOP, id="bishop"),
        pytest.param("a7a8n", chess.KNIGHT, id="knight"),
    ])
    def test_valid_promotion_legal(
        self,
        promotion_position: ChessGame,
        promo_uci: str,
        promo_piece: int
    ) -> None:
        """Pawn promotion to all piece types is legal."""
        legal_moves = promotion_position.get_legal_moves()
        assert chess.Move.from_uci(promo_uci) in legal_moves

    def test_valid_promotion_creates_piece(self, promotion_position: ChessGame) -> None:
        """Pawn promotion creates the promoted piece."""
        promotion_position.make_move(chess.Move.from_uci("a7a8q"))
        assert promotion_position.board.piece_at(chess.A8).piece_type == chess.QUEEN

    def test_valid_promotion_encoding_unique(self, promotion_position: ChessGame) -> None:
        """Move encoding distinguishes promotion types."""
        indices = [
            promotion_position.get_move_index(chess.Move.from_uci("a7a8q")),
            promotion_position.get_move_index(chess.Move.from_uci("a7a8r")),
            promotion_position.get_move_index(chess.Move.from_uci("a7a8b")),
            promotion_position.get_move_index(chess.Move.from_uci("a7a8n")),
        ]
        assert len(indices) == len(set(indices))


class TestChessGameEdgeCases:
    """Tests for edge cases and complex scenarios."""

    # -------------------------------------------------------------------------
    # Draw condition tests
    # -------------------------------------------------------------------------

    def test_edge_threefold_repetition(self) -> None:
        """Threefold repetition allows draw claim."""
        game = ChessGame()

        # Repeat position 3 times
        for _ in range(2):
            game.make_move(chess.Move.from_uci("g1f3"))
            game.make_move(chess.Move.from_uci("g8f6"))
            game.make_move(chess.Move.from_uci("f3g1"))
            game.make_move(chess.Move.from_uci("f6g8"))

        game.make_move(chess.Move.from_uci("g1f3"))
        game.make_move(chess.Move.from_uci("g8f6"))
        game.make_move(chess.Move.from_uci("f3g1"))
        game.make_move(chess.Move.from_uci("f6g8"))

        assert game.board.can_claim_draw()

    def test_edge_fifty_move_rule(self) -> None:
        """Fifty-move rule results in draw."""
        game = ChessGame(fen="7k/8/8/8/8/8/4Q3/4K3 w - - 0 1")

        for _ in range(25):
            game.make_move(chess.Move.from_uci("e2e3"))
            game.make_move(chess.Move.from_uci("h8g8"))
            game.make_move(chess.Move.from_uci("e3e2"))
            game.make_move(chess.Move.from_uci("g8h8"))

        assert game.board.halfmove_clock >= 50
        assert game.is_draw() or game.board.can_claim_draw()

    def test_edge_king_returns_after_castling_cannot_castle(self) -> None:
        """King cannot castle after returning to original square."""
        game = ChessGame(fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

        game.make_move(chess.Move.from_uci("e1g1"))  # White castles
        game.make_move(chess.Move.from_uci("e8g8"))  # Black castles
        game.make_move(chess.Move.from_uci("f2f3"))
        game.make_move(chess.Move.from_uci("f7f6"))
        game.make_move(chess.Move.from_uci("g1f2"))
        game.make_move(chess.Move.from_uci("g8f7"))
        game.make_move(chess.Move.from_uci("f1h1"))
        game.make_move(chess.Move.from_uci("f8h8"))
        game.make_move(chess.Move.from_uci("f2e1"))  # King returns to e1
        game.make_move(chess.Move.from_uci("f7e8"))

        legal_moves = game.get_legal_moves()
        assert chess.Move.from_uci("e1g1") not in legal_moves
        assert chess.Move.from_uci("e1c1") not in legal_moves

    # -------------------------------------------------------------------------
    # Tactical edge cases
    # -------------------------------------------------------------------------

    def test_edge_discovered_check(self) -> None:
        """Discovered check positions are handled correctly."""
        game = ChessGame(fen="3k4/8/8/8/8/8/3B4/3RK3 w - - 0 1")
        legal_moves = game.get_legal_moves()

        # Moving bishop reveals discovered check from rook
        assert chess.Move.from_uci("d2f4") in legal_moves
        assert chess.Move.from_uci("d2g5") in legal_moves

    def test_edge_pinned_piece_cannot_move(self) -> None:
        """Pinned pieces have restricted movement."""
        game = ChessGame(fen="4k3/8/8/8/4r3/8/4N3/4K3 w - - 0 1")
        legal_moves = game.get_legal_moves()

        # Knight on e2 is pinned by rook on e4
        knight_moves = ["e2g1", "e2g3", "e2f4", "e2d4", "e2c3", "e2c1"]
        for move_uci in knight_moves:
            assert chess.Move.from_uci(move_uci) not in legal_moves

    def test_edge_complex_endgame(self) -> None:
        """Complex endgame position is handled correctly."""
        game = ChessGame(fen="8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
        legal_moves = game.get_legal_moves()

        assert len(legal_moves) > 0
        assert not game.is_game_over()

    def test_edge_fen_uses_ascii_only(self) -> None:
        """FEN string uses ASCII characters only."""
        game = ChessGame()
        state = game.get_state()
        assert isinstance(state, str)
        assert all(ord(c) < 128 for c in state)
