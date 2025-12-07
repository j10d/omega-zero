"""
Monte Carlo Tree Search implementation for AlphaZero-style chess.

This module provides the MCTS class for tree search using neural network
evaluation instead of rollouts, following the AlphaZero algorithm.
"""

import numpy as np
import chess

from src.chess_game import ChessGame
from src.neural_network import ChessNN


class Node:
    """
    MCTS tree node.

    Stores statistics for a single game state in the search tree.
    Each node tracks visit counts, accumulated values, and child nodes.
    """

    def __init__(self, prior: float):
        """
        Initialize node with prior probability.

        Args:
            prior: Prior probability P from parent's policy output.
        """
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: dict[chess.Move, "Node"] = {}

    @property
    def q_value(self) -> float:
        """
        Average value Q = W / N.

        Returns:
            Average backed-up value, or 0.0 if unvisited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class MCTS:
    """
    Monte Carlo Tree Search with neural network evaluation.

    Implements AlphaZero-style MCTS without rollouts. Uses a neural network
    to evaluate leaf positions and provide move priors.
    """

    def __init__(
        self,
        neural_network: ChessNN,
        num_simulations: int = 100,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initialize MCTS with configuration.

        Args:
            neural_network: ChessNN for position evaluation and move priors.
            num_simulations: Number of simulations per search (default 100).
            c_puct: Exploration constant for PUCT formula (default 1.0).
            dirichlet_alpha: Dirichlet noise shape parameter (default 0.3).
            dirichlet_epsilon: Noise mixing ratio (default 0.25).
        """
        self.neural_network = neural_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(
        self,
        game: ChessGame,
        temperature: float = 1.0,
        add_noise: bool = False
    ) -> chess.Move:
        """
        Run MCTS and return selected move.

        Args:
            game: Current game state.
            temperature: Move selection temperature (0 = greedy, 1 = proportional).
            add_noise: Whether to add Dirichlet noise to root priors.

        Returns:
            Selected chess.Move.

        Raises:
            ValueError: If game is already over.
        """
        if game.is_game_over():
            raise ValueError("Cannot search: game is already over")

        # Build tree from root
        root = self._build_tree(game, add_noise)

        # Select move based on visit counts
        return self._select_move(root, temperature)

    def get_policy(
        self,
        game: ChessGame,
        temperature: float = 1.0,
        add_noise: bool = False
    ) -> tuple[chess.Move, np.ndarray]:
        """
        Run MCTS and return selected move with policy vector.

        Args:
            game: Current game state.
            temperature: Move selection temperature.
            add_noise: Whether to add Dirichlet noise to root priors.

        Returns:
            Tuple of (selected move, policy vector of shape (4672,)).

        Raises:
            ValueError: If game is already over.
        """
        if game.is_game_over():
            raise ValueError("Cannot search: game is already over")

        # Build tree from root
        root = self._build_tree(game, add_noise)

        # Build policy vector from visit counts
        policy = np.zeros(ChessGame.POLICY_SIZE, dtype=np.float32)
        for move, child in root.children.items():
            index = game.get_move_index(move)
            policy[index] = child.visit_count

        # Normalize
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum

        # Select move
        move = self._select_move(root, temperature)

        return move, policy

    def _build_tree(self, game: ChessGame, add_noise: bool) -> Node:
        """
        Build search tree by running simulations.

        Args:
            game: Game state for root.
            add_noise: Whether to add Dirichlet noise to root.

        Returns:
            Root node of search tree.
        """
        # Create root node
        root = Node(prior=1.0)

        # Expand root immediately
        self._expand_node(root, game)

        # Add Dirichlet noise to root if requested
        if add_noise:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            self._run_simulation(root, game.clone())

        return root

    def _run_simulation(self, root: Node, game: ChessGame) -> None:
        """
        Run single simulation: select -> expand -> evaluate -> backup.

        Args:
            root: Root node of search tree.
            game: Cloned game state (will be modified during traversal).
        """
        # Track path for backup
        path = [root]
        node = root

        # SELECT: traverse tree using PUCT until leaf
        while node.children:
            move, node = self._select_child(node)
            game.make_move(move)
            path.append(node)

        # EVALUATE
        if game.is_game_over():
            # Terminal node: use actual game result
            value = game.get_result()
            if value is None:
                value = 0.0
        else:
            # Non-terminal: expand and use neural network value
            self._expand_node(node, game)
            board_tensor = game.get_canonical_board()
            _, value = self.neural_network.predict(board_tensor)
            value = float(value)

        # BACKUP: propagate value with alternating signs
        self._backup(path, value)

    def _select_child(self, node: Node) -> tuple[chess.Move, Node]:
        """
        Select child with highest PUCT score.

        Args:
            node: Parent node to select from.

        Returns:
            Tuple of (selected move, selected child node).
        """
        best_score = float("-inf")
        best_move = None
        best_child = None

        sqrt_parent_visits = np.sqrt(node.visit_count)

        for move, child in node.children.items():
            # PUCT = -Q(child) + c_puct * P(child) * sqrt(N_parent) / (1 + N_child)
            # Negate Q because opponent's gain is our loss
            exploration = self.c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = -child.q_value + exploration

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _expand_node(self, node: Node, game: ChessGame) -> None:
        """
        Expand node by creating children for all legal moves.

        Args:
            node: Node to expand.
            game: Game state at this node.
        """
        if game.is_game_over():
            return

        # Get neural network policy
        board_tensor = game.get_canonical_board()
        policy, _ = self.neural_network.predict(board_tensor)

        # Get legal moves mask
        legal_mask = game.get_legal_moves_mask()

        # Mask and normalize policy
        masked_policy = policy * legal_mask
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy = masked_policy / policy_sum
        else:
            # Fallback: uniform over legal moves
            masked_policy = legal_mask.astype(np.float32)
            masked_policy = masked_policy / masked_policy.sum()

        # Create child nodes
        for move in game.get_legal_moves():
            index = game.get_move_index(move)
            prior = masked_policy[index]
            node.children[move] = Node(prior=prior)

    def _backup(self, path: list[Node], value: float) -> None:
        """
        Backup value through path with alternating signs.

        Value is from the perspective of the player at the leaf.
        Each step up the tree flips perspective.

        Args:
            path: List of nodes from root to leaf.
            value: Value to back up (from leaf's perspective).
        """
        # Start from leaf and work backwards
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # Flip sign for next level

    def _add_dirichlet_noise(self, node: Node) -> None:
        """
        Add Dirichlet noise to root node priors.

        Args:
            node: Root node.
        """
        if not node.children:
            return

        num_children = len(node.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_children)

        for i, child in enumerate(node.children.values()):
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * noise[i]

    def _select_move(self, root: Node, temperature: float) -> chess.Move:
        """
        Select move from root based on visit counts.

        Args:
            root: Root node of search tree.
            temperature: Selection temperature (0 = greedy).

        Returns:
            Selected move.
        """
        moves = list(root.children.keys())
        visit_counts = np.array([root.children[m].visit_count for m in moves])

        if temperature == 0:
            # Greedy: select most visited
            best_idx = np.argmax(visit_counts)
            return moves[best_idx]
        else:
            # Stochastic: sample proportional to visit_count^(1/temperature)
            visit_counts = visit_counts.astype(np.float64)
            if visit_counts.sum() == 0:
                # Fallback: uniform random
                return np.random.choice(moves)

            # Apply temperature
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / probs.sum()

            # Sample
            idx = np.random.choice(len(moves), p=probs)
            return moves[idx]
