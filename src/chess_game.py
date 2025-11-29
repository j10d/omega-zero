"""
ChessGame class - Game Environment & Rules Engine (STUB).

This is a stub implementation with method signatures only.
All methods raise NotImplementedError to verify tests fail properly.
"""

import numpy as np
import chess


class ChessGame:
    """
    Wrapper around python-chess providing AlphaZero-specific interfaces.

    This class provides a thin wrapper around python-chess that handles:
    - Game state management
    - Move execution and validation
    - Neural network interface (canonical board, move encoding)
    - Game status detection (checkmate, stalemate, draws)
    """

    def __init__(self, fen: str | None = None):
        """
        Initialize chess game.

        Args:
            fen: FEN string for initial position. None for starting position.
        """
        raise NotImplementedError

    def clone(self) -> "ChessGame":
        """
        Create independent copy of game state.

        Returns:
            New ChessGame instance with identical state.
        """
        raise NotImplementedError

    def get_state(self) -> str:
        """
        Get current game state as FEN string.

        Returns:
            FEN string representing current position.
        """
        raise NotImplementedError

    def get_legal_moves(self) -> list[chess.Move]:
        """
        Get all legal moves from current position.

        Returns:
            List of legal chess.Move objects.
        """
        raise NotImplementedError

    def get_legal_moves_mask(self) -> np.ndarray:
        """
        Get boolean mask of legal moves.

        Returns:
            Boolean array of shape (4672,) where True indicates legal move.
        """
        raise NotImplementedError

    def make_move(self, move: chess.Move) -> None:
        """
        Execute a move on the board.

        Args:
            move: The chess.Move to execute.

        Raises:
            ValueError: If move is illegal.
        """
        raise NotImplementedError

    def undo_move(self) -> None:
        """
        Undo the last move.

        Raises:
            ValueError: If no moves to undo.
        """
        raise NotImplementedError

    def is_game_over(self) -> bool:
        """
        Check if game has ended.

        Returns:
            True if game is over (checkmate, stalemate, or draw).
        """
        raise NotImplementedError

    def get_result(self) -> float | None:
        """
        Get game result from current player's perspective.

        Returns:
            +1.0 if current player won
            -1.0 if current player lost
            0.0 if draw
            None if game not over
        """
        raise NotImplementedError

    def is_draw(self) -> bool:
        """
        Check if position is a draw.

        Returns:
            True if position is drawn (stalemate, insufficient material, etc.).
        """
        raise NotImplementedError

    def get_canonical_board(self) -> np.ndarray:
        """
        Get board representation from current player's perspective.

        Returns canonical form where board is always shown from perspective
        of player to move (flips board if black to move).

        Returns:
            Array of shape (8, 8, 14) with dtype float32.
            Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
            Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
            Plane 12: Repetition count
            Plane 13: En passant square
        """
        raise NotImplementedError

    def get_move_index(self, move: chess.Move) -> int:
        """
        Convert chess.Move to policy network index.

        Args:
            move: The chess move to encode.

        Returns:
            Integer index in range [0, 4671].

        Raises:
            ValueError: If move is illegal in current position.
        """
        raise NotImplementedError

    def get_move_from_index(self, index: int) -> chess.Move:
        """
        Convert policy network index to chess.Move.

        Args:
            index: Integer in range [0, 4671].

        Returns:
            The corresponding chess.Move.

        Raises:
            ValueError: If index doesn't correspond to legal move.
        """
        raise NotImplementedError
