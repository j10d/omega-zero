# MCTS Specification

## Overview

Monte Carlo Tree Search using neural network for position evaluation and move priors. Implements the AlphaZero-style MCTS without rollouts.

## Design Decisions

- **Tree structure:** Object-based (Node class with children dictionary)
- **Transposition handling:** None (each path independent)
- **Parallelization:** None for v1.0 (single-threaded)
- **Leaf evaluation:** Neural network value head (no rollouts)

## Node Class

```python
class Node:
    def __init__(self, prior: float):
        self.prior = prior           # P: from parent's policy output
        self.visit_count = 0         # N: number of visits
        self.total_value = 0.0       # W: sum of values backed up
        self.children = {}           # dict[chess.Move, Node]
    
    @property
    def q_value(self) -> float:
        """Q = W / N, or 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
```

## PUCT Formula

Used to select which child to explore:

```
PUCT(child) = Q(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N(child))
```

Where:
- `Q(child)` = average value from child's perspective (negated from parent's view)
- `P(child)` = prior probability from neural network
- `N_parent` = visit count of parent node
- `N(child)` = visit count of child node
- `c_puct` = exploration constant (default 1.0)

**Sign convention:** Q values are stored from each node's perspective. When computing PUCT at the parent, negate child's Q (opponent's gain is our loss).

## Search Algorithm

```
repeat for num_simulations:
    1. SELECT
       - Start at root
       - While node has children:
           - Pick child with highest PUCT score
           - Descend to that child
       - Stop at leaf (no children)
    
    2. EXPAND (if not terminal)
       - Get neural network policy and value for leaf position
       - Create child nodes for each legal move
       - Set each child's prior from masked/normalized policy
    
    3. EVALUATE
       - If terminal: use actual game result (+1, -1, 0)
       - If non-terminal: use neural network value output
    
    4. BACKUP
       - Walk back up the path to root
       - At each node: N += 1, W += value
       - Flip value sign at each level (alternating perspectives)
```

## Move Selection

After search completes, select move from root based on visit counts:

**Temperature = 0 (greedy):**
```python
move = argmax(child.visit_count for child in root.children)
```

**Temperature > 0 (stochastic):**
```python
counts = [child.visit_count ** (1 / temperature) for child in root.children]
probs = counts / sum(counts)
move = sample(moves, weights=probs)
```

## Dirichlet Noise

Optional noise added to root priors for exploration during self-play:

```python
noise = np.random.dirichlet([alpha] * num_legal_moves)
child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
```

Parameters:
- `alpha = 0.3` (chess-specific, ~10/average_legal_moves)
- `epsilon = 0.25` (25% noise, 75% network prior)

Applied only at root, only during self-play training.

## Class Interface

```python
class MCTS:
    def __init__(
        self,
        neural_network: ChessNN,
        num_simulations: int = 100,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    )
    
    def search(
        self,
        game: ChessGame,
        temperature: float = 1.0,
        add_noise: bool = False
    ) -> chess.Move
    
    def get_policy(
        self,
        game: ChessGame,
        temperature: float = 1.0,
        add_noise: bool = False
    ) -> tuple[chess.Move, np.ndarray]
```

## Method Specifications

### `__init__()`

Store configuration. Does not build tree (that happens per search).

### `search()`

Run MCTS and return selected move.

**Parameters:**
- `game`: Current game state
- `temperature`: Move selection temperature (0 = greedy, 1 = proportional)
- `add_noise`: Whether to add Dirichlet noise to root priors

**Returns:** Selected `chess.Move`

**Raises:** `ValueError` if game is already over

### `get_policy()`

Run MCTS and return both the selected move and the full policy vector.

**Parameters:** Same as `search()`

**Returns:** 
- `move`: Selected `chess.Move`
- `policy`: `np.ndarray` of shape `(4672,)` with visit count distribution

The policy vector is needed for training data. It represents the improved policy from MCTS (better than raw network output).

**Policy construction:**
```python
policy = np.zeros(4672)
for move, child in root.children.items():
    policy[game.get_move_index(move)] = child.visit_count
policy = policy / policy.sum()  # Normalize
```

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| num_simulations | 100 | Simulations per search |
| c_puct | 1.0 | Exploration constant |
| dirichlet_alpha | 0.3 | Noise shape (chess-specific) |
| dirichlet_epsilon | 0.25 | Noise mixing ratio |
| temperature | 1.0 | Move selection (caller provides) |

## Implementation Notes

- Root node prior is irrelevant (set to 1.0 by convention)
- Root starts with visit_count = 0, gets incremented during first backup
- Terminal positions: don't expand, return actual result
- Neural network must be built before search
- Game is cloned internally for tree traversal (original unchanged)

## Usage Example

```python
from chess_game import ChessGame
from neural_network import ChessNN
from mcts import MCTS

# Setup
nn = ChessNN()
nn.load_model("trained_model.keras")
mcts = MCTS(neural_network=nn, num_simulations=100)

# Play a move (greedy)
game = ChessGame()
move = mcts.search(game, temperature=0)
game.make_move(move)

# For training data (with exploration)
game = ChessGame()
move, policy = mcts.get_policy(game, temperature=1.0, add_noise=True)
# policy is the MCTS-improved target for training
```

## Self-Play Temperature Schedule

Not implemented in MCTS class. Caller (self-play engine) manages:

```python
for move_number in range(max_moves):
    temp = 1.0 if move_number < 30 else 0.0
    move, policy = mcts.get_policy(game, temperature=temp, add_noise=True)
    # Store (position, policy, result) for training
    game.make_move(move)
```
