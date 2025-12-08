# Board Representation Specification

## Overview

The canonical board representation is an `(8, 8, 18)` tensor of dtype `float32`. This tensor encodes the complete game state needed for neural network evaluation.

## Canonical Perspective

The board is always represented from the **current player's perspective**:
- Current player's pieces appear in planes 0-5
- Opponent's pieces appear in planes 6-11
- When black to move, ranks are flipped so current player's pieces appear at the bottom
- Files are never flipped (a-file stays a-file)

This allows the neural network to learn "my pieces vs opponent pieces" rather than "white vs black".

## Tensor Layout

| Plane | Description | Values |
|-------|-------------|--------|
| 0 | Current player pawns | 1.0 where piece exists |
| 1 | Current player knights | 1.0 where piece exists |
| 2 | Current player bishops | 1.0 where piece exists |
| 3 | Current player rooks | 1.0 where piece exists |
| 4 | Current player queens | 1.0 where piece exists |
| 5 | Current player king | 1.0 where piece exists |
| 6 | Opponent pawns | 1.0 where piece exists |
| 7 | Opponent knights | 1.0 where piece exists |
| 8 | Opponent bishops | 1.0 where piece exists |
| 9 | Opponent rooks | 1.0 where piece exists |
| 10 | Opponent queens | 1.0 where piece exists |
| 11 | Opponent king | 1.0 where piece exists |
| 12 | Repetition count | Entire plane filled (see below) |
| 13 | En passant square | 1.0 at target square |
| 14 | Current player kingside castling | Entire plane: 1.0 if available, 0.0 if not |
| 15 | Current player queenside castling | Entire plane: 1.0 if available, 0.0 if not |
| 16 | Opponent kingside castling | Entire plane: 1.0 if available, 0.0 if not |
| 17 | Opponent queenside castling | Entire plane: 1.0 if available, 0.0 if not |

## Array Indexing

```
board_array[rank_index, file_index, plane]
```

- `rank_index`: 0 = rank 8 (top), 7 = rank 1 (bottom) before perspective flip
- `file_index`: 0 = a-file, 7 = h-file (never flipped)
- After perspective flip for black: rank indices are reversed

## Special State Encoding

### Repetition Count (Plane 12)

Entire plane filled with a single normalized value:

| Occurrences | Value |
|-------------|-------|
| 1 (no repetition) | 0.333 |
| 2 (repeated once) | 0.667 |
| 3+ (threefold) | 1.0 |

### En Passant (Plane 13)

- If en passant capture is available: 1.0 at the target square (the square the capturing pawn moves to)
- If no en passant: entire plane is 0.0
- Square position respects perspective flip

### Castling Rights (Planes 14-17)

Binary planes filled entirely with 0.0 or 1.0:
- 1.0 = castling right is available
- 0.0 = castling right is not available

"Current player" and "opponent" are determined by whose turn it is, matching the canonical perspective.

## Implementation Notes

- Method: `ChessGame.get_canonical_board() -> np.ndarray`
- Shape assertion: `assert board.shape == (8, 8, 18)`
- All values in range [0.0, 1.0]
- Piece planes are mutually exclusive (each square has at most one piece)

## Example: Starting Position (White to Move)

```
Plane 0 (white pawns): 1.0 at rank_index=6 (rank 2), files 0-7
Plane 5 (white king): 1.0 at rank_index=7, file_index=4 (e1)
Plane 11 (black king): 1.0 at rank_index=0, file_index=4 (e8)
Plane 12 (repetition): 0.333 everywhere (first occurrence)
Plane 13 (en passant): 0.0 everywhere
Planes 14-17 (castling): 1.0 everywhere (all castling available)
```
