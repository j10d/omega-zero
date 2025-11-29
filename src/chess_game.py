"""
ChessGame class - Game Environment & Rules Engine.

Thin wrapper around python-chess providing AlphaZero-specific interfaces
for neural network input/output, MCTS tree search, and self-play generation.
"""

import numpy as np
import chess


class ChessGame:
    """
    Chess game environment wrapper for AlphaZero-style reinforcement learning.

    This class provides a thin wrapper around python-chess that handles:
    - Game state management
    - Move execution and validation
    - Neural network interface (canonical board, move encoding)
    - Game status detection (checkmate, stalemate, draws)

    The python-chess library handles all chess rules, legal move generation,
    and game state. This wrapper focuses on providing the interfaces needed
    for neural network training and MCTS.
    """

    # Move encoding constants
    # Using simplified encoding: from_square * 64 + to_square for base moves
    # Promotions use additional indices to distinguish piece types
    POLICY_SIZE = 4672  # Total policy vector size (padded for compatibility)
    BASE_MOVES = 4096   # 64 * 64 = all from-to combinations

    def __init__(self, fen: str | None = None):
        """
        Initialize chess game.

        Args:
            fen: FEN string for initial position. If None, uses standard
                starting position.
        """
        if fen is None:
            self.board = chess.Board()
        else:
            self.board = chess.Board(fen)

    def clone(self) -> "ChessGame":
        """
        Create independent copy of game state.

        Creates a new ChessGame instance with identical board state.
        Modifications to the clone will not affect the original.

        Returns:
            New ChessGame instance with identical state.
        """
        clone = ChessGame.__new__(ChessGame)
        clone.board = self.board.copy()
        return clone

    def get_state(self) -> str:
        """
        Get current game state as FEN string.

        Returns:
            FEN string representing current position, including turn,
            castling rights, en passant square, and move counters.
        """
        return self.board.fen()

    def get_legal_moves(self) -> list[chess.Move]:
        """
        Get all legal moves from current position.

        Returns:
            List of legal chess.Move objects in current position.
        """
        return list(self.board.legal_moves)

    def get_legal_moves_mask(self) -> np.ndarray:
        """
        Get boolean mask of legal moves.

        Creates a boolean array where True at index i means the move
        corresponding to index i is legal in the current position.

        Returns:
            Boolean array of shape (4672,) where True indicates legal move.
        """
        mask = np.zeros(self.POLICY_SIZE, dtype=bool)
        for move in self.board.legal_moves:
            index = self.get_move_index(move)
            mask[index] = True
        return mask

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

    def undo_move(self) -> None:
        """
        Undo the last move.

        Restores board state to before the last move was made.

        Raises:
            IndexError: If no moves to undo (handled by python-chess).
        """
        self.board.pop()

    def is_game_over(self) -> bool:
        """
        Check if game has ended.

        Game is over if:
        - Checkmate
        - Stalemate
        - Insufficient material
        - Fifty-move rule
        - Threefold repetition

        Returns:
            True if game is over by any condition.
        """
        return self.board.is_game_over()

    def get_result(self) -> float | None:
        """
        Get game result from current player's perspective.

        Returns result relative to the player whose turn it currently is,
        not necessarily the player who just moved.

        Returns:
            +1.0 if current player won
            -1.0 if current player lost
            0.0 if draw
            None if game not over
        """
        if not self.board.is_game_over():
            return None

        # Get result from board (returns "1-0", "0-1", "1/2-1/2", or "*")
        result = self.board.result()

        # Draw
        if result == "1/2-1/2":
            return 0.0

        # Checkmate - current player has no legal moves and is in check
        # If current player is in checkmate, they lost
        if self.board.is_checkmate():
            return -1.0

        # Stalemate or other draw condition
        return 0.0

    def is_draw(self) -> bool:
        """
        Check if position is a draw.

        Checks for:
        - Stalemate (no legal moves, not in check)
        - Insufficient material
        - Fifty-move rule
        - Threefold repetition
        - Fivefold repetition
        - Seventy-five move rule

        Returns:
            True if position is drawn by any rule.
        """
        return (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.is_fifty_moves()
            or self.board.is_repetition()
        )

    def get_canonical_board(self) -> np.ndarray:
        """
        Get board representation from current player's perspective.

        Returns canonical form where board is always shown from perspective
        of player to move. If black to move, board is flipped so black pieces
        are at bottom.

        This allows the neural network to learn "my pieces vs opponent pieces"
        instead of "white vs black", reducing what it needs to learn.

        Returns:
            Array of shape (8, 8, 14) with dtype float32.
            - Planes 0-5: Current player's pieces (P, N, B, R, Q, K)
            - Planes 6-11: Opponent's pieces (P, N, B, R, Q, K)
            - Plane 12: Repetition count (normalized)
            - Plane 13: En passant square
        """
        board_array = np.zeros((8, 8, 14), dtype=np.float32)

        # Determine if we need to flip (black to move)
        flip = self.board.turn == chess.BLACK

        # Piece type mapping for both colors
        # Current player's pieces go in planes 0-5
        # Opponent's pieces go in planes 6-11
        current_color = self.board.turn
        opponent_color = not current_color

        # Map piece types to plane indices
        piece_to_plane = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        # Fill piece planes
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                # Get rank and file from chess square
                # chess.square_rank() returns 0-7 where 0=rank 1, 7=rank 8
                rank = chess.square_rank(square)
                file = chess.square_file(square)

                # Convert to array indices where index 0 = rank 8, index 7 = rank 1
                array_rank = 7 - rank

                # Flip if black to move (flip entire board)
                if flip:
                    array_rank = 7 - array_rank
                    file = 7 - file

                # Determine which plane
                plane_offset = 0 if piece.color == current_color else 6
                plane = plane_offset + piece_to_plane[piece.piece_type]

                board_array[array_rank, file, plane] = 1.0

        # Plane 12: Repetition count (number of times position has occurred)
        # Normalize by dividing by 3 (threefold repetition is max meaningful)
        repetition_count = sum(
            1 for _ in self.board.legal_moves
        )  # Placeholder - python-chess doesn't expose this easily
        # Use a simple heuristic: if can claim draw by repetition, set to 1
        if self.board.can_claim_draw():
            board_array[:, :, 12] = 1.0

        # Plane 13: En passant square
        if self.board.ep_square is not None:
            ep_rank = chess.square_rank(self.board.ep_square)
            ep_file = chess.square_file(self.board.ep_square)

            # Convert to array indices
            ep_array_rank = 7 - ep_rank

            if flip:
                ep_array_rank = 7 - ep_array_rank
                ep_file = 7 - ep_file

            board_array[ep_array_rank, ep_file, 13] = 1.0

        return board_array

    def get_move_index(self, move: chess.Move) -> int:
        """
        Convert chess.Move to policy network index.

        Uses simplified encoding:
        - Regular moves: from_square * 64 + to_square
        - Promotions: 4096 + (from_square * 4 + to_square % 8) * 3 + promo_type
          where promo_type: 0=knight, 1=bishop, 2=rook (queen uses base encoding)

        Args:
            move: The chess move to encode.

        Returns:
            Integer index in range [0, 4671].

        Raises:
            ValueError: If move is illegal in current position.
        """
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move.uci()}")

        from_square = move.from_square
        to_square = move.to_square

        # Handle promotions specially
        if move.promotion is not None:
            # Queen promotion uses base encoding (most common)
            if move.promotion == chess.QUEEN:
                return from_square * 64 + to_square

            # Underpromotions get special indices
            # Base index for underpromotions
            base = self.BASE_MOVES

            # Calculate offset for this pawn's file and promotion type
            from_file = chess.square_file(from_square)
            to_file = chess.square_file(to_square)

            # Promotion type offset (0=knight, 1=bishop, 2=rook)
            promo_map = {
                chess.KNIGHT: 0,
                chess.BISHOP: 1,
                chess.ROOK: 2,
            }
            promo_offset = promo_map[move.promotion]

            # Index: base + (from_file * 3 * 3 + (to_file - from_file + 1) * 3 + promo)
            # This gives unique index for each underpromotion
            # 8 files * 3 directions * 3 piece types = 72 underpromotions max
            direction = to_file - from_file + 1  # -1, 0, or +1 mapped to 0, 1, 2
            index = base + from_file * 9 + direction * 3 + promo_offset

            return index

        # Regular move (including queen promotions)
        return from_square * 64 + to_square

    def get_move_from_index(self, index: int) -> chess.Move:
        """
        Convert policy network index to chess.Move.

        Reverses the encoding from get_move_index().

        Args:
            index: Integer in range [0, 4671].

        Returns:
            The corresponding chess.Move.

        Raises:
            ValueError: If index doesn't correspond to legal move in current position.
        """
        if index < 0 or index >= self.POLICY_SIZE:
            raise ValueError(f"Index {index} out of range [0, {self.POLICY_SIZE})")

        # Check if this is an underpromotion index
        if index >= self.BASE_MOVES:
            # Decode underpromotion
            offset = index - self.BASE_MOVES
            from_file = offset // 9
            remainder = offset % 9
            direction = remainder // 3  # 0, 1, or 2
            promo_offset = remainder % 3

            # Map back to piece type
            promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
            promotion = promo_pieces[promo_offset]

            # Find the actual move with this from_file and promotion
            # We need to search legal moves since we don't know the exact squares
            for move in self.board.legal_moves:
                if (
                    move.promotion == promotion
                    and chess.square_file(move.from_square) == from_file
                ):
                    to_file = chess.square_file(move.to_square)
                    if to_file - from_file + 1 == direction:
                        return move

            raise ValueError(f"No legal move found for underpromotion index {index}")

        # Regular move
        from_square = index // 64
        to_square = index % 64

        # Find this move in legal moves (might be a queen promotion)
        for move in self.board.legal_moves:
            if move.from_square == from_square and move.to_square == to_square:
                return move

        raise ValueError(f"No legal move found for index {index}")
