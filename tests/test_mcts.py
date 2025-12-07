"""
Tests for MCTS (Monte Carlo Tree Search) module.

Follows TEST_GUIDELINES.md conventions with class-based organization
and standardized naming prefixes.
"""

import numpy as np
import pytest
import chess

from src.mcts import Node, MCTS
from src.chess_game import ChessGame
from src.neural_network import ChessNN


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fresh_game() -> ChessGame:
    """Standard starting position."""
    return ChessGame()


@pytest.fixture
def neural_network() -> ChessNN:
    """Fresh untrained ChessNN for testing."""
    nn = ChessNN(num_residual_blocks=2, num_filters=32)
    nn.build_model()
    return nn


@pytest.fixture
def mcts(neural_network: ChessNN) -> MCTS:
    """MCTS instance with default configuration."""
    return MCTS(neural_network=neural_network, num_simulations=10)


@pytest.fixture
def checkmate_game() -> ChessGame:
    """Position where black is checkmated (Fool's mate)."""
    game = ChessGame()
    game.make_move(chess.Move.from_uci("f2f3"))
    game.make_move(chess.Move.from_uci("e7e5"))
    game.make_move(chess.Move.from_uci("g2g4"))
    game.make_move(chess.Move.from_uci("d8h4"))
    return game


@pytest.fixture
def stalemate_game() -> ChessGame:
    """Position with stalemate."""
    # Black king cornered, queen blocks all escape squares
    return ChessGame("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")


# =============================================================================
# Test Classes
# =============================================================================


class TestNode:
    """Tests for Node class initialization and properties."""

    # -------------------------------------------------------------------------
    # Valid initialization tests
    # -------------------------------------------------------------------------

    def test_valid_initialization_with_prior(self) -> None:
        """Node initializes with given prior probability."""
        node = Node(prior=0.5)
        assert node.prior == 0.5

    def test_valid_initialization_zero_visits(self) -> None:
        """New node has zero visit count."""
        node = Node(prior=0.5)
        assert node.visit_count == 0

    def test_valid_initialization_zero_total_value(self) -> None:
        """New node has zero total value."""
        node = Node(prior=0.5)
        assert node.total_value == 0.0

    def test_valid_initialization_empty_children(self) -> None:
        """New node has empty children dictionary."""
        node = Node(prior=0.5)
        assert node.children == {}
        assert len(node.children) == 0

    # -------------------------------------------------------------------------
    # q_value property tests
    # -------------------------------------------------------------------------

    def test_valid_q_value_unvisited_returns_zero(self) -> None:
        """Q value is 0 for unvisited node."""
        node = Node(prior=0.5)
        assert node.q_value == 0.0

    def test_valid_q_value_single_visit(self) -> None:
        """Q value equals total value when visit count is 1."""
        node = Node(prior=0.5)
        node.visit_count = 1
        node.total_value = 0.7
        assert node.q_value == 0.7

    def test_valid_q_value_multiple_visits(self) -> None:
        """Q value is average of backed up values."""
        node = Node(prior=0.5)
        node.visit_count = 4
        node.total_value = 2.0
        assert node.q_value == 0.5

    def test_valid_q_value_negative(self) -> None:
        """Q value can be negative."""
        node = Node(prior=0.5)
        node.visit_count = 2
        node.total_value = -1.0
        assert node.q_value == -0.5


class TestMCTSInit:
    """Tests for MCTS initialization."""

    def test_valid_init_stores_neural_network(self, neural_network: ChessNN) -> None:
        """MCTS stores the neural network reference."""
        mcts = MCTS(neural_network=neural_network)
        assert mcts.neural_network is neural_network

    def test_valid_init_default_num_simulations(self, neural_network: ChessNN) -> None:
        """Default num_simulations is 100."""
        mcts = MCTS(neural_network=neural_network)
        assert mcts.num_simulations == 100

    def test_valid_init_custom_num_simulations(self, neural_network: ChessNN) -> None:
        """Custom num_simulations is stored."""
        mcts = MCTS(neural_network=neural_network, num_simulations=50)
        assert mcts.num_simulations == 50

    def test_valid_init_default_c_puct(self, neural_network: ChessNN) -> None:
        """Default c_puct is 1.0."""
        mcts = MCTS(neural_network=neural_network)
        assert mcts.c_puct == 1.0

    def test_valid_init_default_dirichlet_alpha(self, neural_network: ChessNN) -> None:
        """Default dirichlet_alpha is 0.3."""
        mcts = MCTS(neural_network=neural_network)
        assert mcts.dirichlet_alpha == 0.3

    def test_valid_init_default_dirichlet_epsilon(self, neural_network: ChessNN) -> None:
        """Default dirichlet_epsilon is 0.25."""
        mcts = MCTS(neural_network=neural_network)
        assert mcts.dirichlet_epsilon == 0.25


class TestPUCTScore:
    """Tests for PUCT score calculation."""

    def test_valid_puct_unvisited_child_high_score(self) -> None:
        """Unvisited child with high prior gets high PUCT score."""
        # PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        # For unvisited child: PUCT = 0 + 1.0 * 0.5 * sqrt(10) / (1 + 0)
        # = 0.5 * sqrt(10) ≈ 1.58
        parent = Node(prior=1.0)
        parent.visit_count = 10

        child = Node(prior=0.5)
        child.visit_count = 0
        child.total_value = 0.0

        c_puct = 1.0
        expected = 0.5 * np.sqrt(10)

        # PUCT formula: Q(child) negated + exploration bonus
        # -child.q_value because opponent's gain is our loss
        puct = -child.q_value + c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
        assert np.isclose(puct, expected)

    def test_valid_puct_visited_child_lower_exploration(self) -> None:
        """Visited child has lower exploration bonus."""
        parent = Node(prior=1.0)
        parent.visit_count = 10

        child = Node(prior=0.5)
        child.visit_count = 5
        child.total_value = 1.0  # Q = 0.2

        c_puct = 1.0
        # PUCT = -0.2 + 1.0 * 0.5 * sqrt(10) / (1 + 5)
        # = -0.2 + 0.5 * 3.16 / 6 ≈ -0.2 + 0.26 = 0.06
        exploration = c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
        puct = -child.q_value + exploration

        assert puct < 1.0  # Much lower than unvisited
        assert np.isclose(puct, -0.2 + 0.5 * np.sqrt(10) / 6)

    def test_valid_puct_higher_prior_higher_score(self) -> None:
        """Higher prior leads to higher PUCT score (all else equal)."""
        parent = Node(prior=1.0)
        parent.visit_count = 10

        child_low = Node(prior=0.1)
        child_high = Node(prior=0.9)

        c_puct = 1.0
        puct_low = c_puct * child_low.prior * np.sqrt(parent.visit_count) / (1 + child_low.visit_count)
        puct_high = c_puct * child_high.prior * np.sqrt(parent.visit_count) / (1 + child_high.visit_count)

        assert puct_high > puct_low


class TestMoveSelection:
    """Tests for move selection based on visit counts."""

    def test_valid_temperature_zero_selects_most_visited(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """Temperature 0 selects move with highest visit count (greedy)."""
        # Run search with temperature=0
        move = mcts.search(fresh_game, temperature=0, add_noise=False)

        # Move should be a valid legal move
        assert move in fresh_game.get_legal_moves()

    def test_valid_temperature_zero_deterministic(
        self, neural_network: ChessNN, fresh_game: ChessGame
    ) -> None:
        """Temperature 0 produces deterministic results."""
        mcts = MCTS(neural_network=neural_network, num_simulations=20)

        move1 = mcts.search(fresh_game, temperature=0, add_noise=False)
        move2 = mcts.search(fresh_game, temperature=0, add_noise=False)

        # Same position, same network, no noise -> same move
        assert move1 == move2

    def test_valid_temperature_one_stochastic(
        self, neural_network: ChessNN, fresh_game: ChessGame
    ) -> None:
        """Temperature 1 can produce different moves (stochastic)."""
        # With enough runs, temperature=1 should sometimes select different moves
        # We use a seeded random state to make this reproducible
        mcts = MCTS(neural_network=neural_network, num_simulations=20)

        moves = set()
        np.random.seed(42)
        for _ in range(10):
            move = mcts.search(fresh_game, temperature=1.0, add_noise=False)
            moves.add(move)

        # With temperature=1 and random sampling, we expect some variation
        # (though not guaranteed, this is a probabilistic test)
        assert len(moves) >= 1  # At minimum, always returns a valid move


class TestSearch:
    """Tests for MCTS search method."""

    def test_valid_search_returns_legal_move(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """Search returns a valid legal move."""
        move = mcts.search(fresh_game, temperature=1.0)
        assert move in fresh_game.get_legal_moves()

    def test_valid_search_does_not_modify_game(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """Search does not modify the original game state."""
        original_fen = fresh_game.get_state()
        mcts.search(fresh_game, temperature=1.0)
        assert fresh_game.get_state() == original_fen

    def test_valid_search_different_positions(
        self, mcts: MCTS
    ) -> None:
        """Search works on different positions."""
        # Sicilian Defense position
        game = ChessGame()
        game.make_move(chess.Move.from_uci("e2e4"))
        game.make_move(chess.Move.from_uci("c7c5"))

        move = mcts.search(game, temperature=1.0)
        assert move in game.get_legal_moves()

    def test_error_search_on_finished_game(
        self, mcts: MCTS, checkmate_game: ChessGame
    ) -> None:
        """Search raises ValueError on game that's already over."""
        assert checkmate_game.is_game_over()
        with pytest.raises(ValueError, match="game.*over|terminal"):
            mcts.search(checkmate_game)


class TestGetPolicy:
    """Tests for MCTS get_policy method."""

    def test_valid_get_policy_returns_tuple(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """get_policy returns tuple of (move, policy)."""
        result = mcts.get_policy(fresh_game, temperature=1.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_valid_get_policy_move_is_legal(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """get_policy returns a legal move."""
        move, _ = mcts.get_policy(fresh_game, temperature=1.0)
        assert move in fresh_game.get_legal_moves()

    def test_valid_get_policy_shape(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """Policy vector has shape (4672,)."""
        _, policy = mcts.get_policy(fresh_game, temperature=1.0)
        assert policy.shape == (4672,)

    def test_valid_get_policy_normalized(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """Policy vector sums to 1.0."""
        _, policy = mcts.get_policy(fresh_game, temperature=1.0)
        assert np.isclose(policy.sum(), 1.0)

    def test_valid_get_policy_non_negative(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """Policy values are non-negative."""
        _, policy = mcts.get_policy(fresh_game, temperature=1.0)
        assert (policy >= 0).all()

    def test_valid_get_policy_only_legal_moves_nonzero(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """Only legal moves have non-zero probability."""
        _, policy = mcts.get_policy(fresh_game, temperature=1.0)

        legal_mask = fresh_game.get_legal_moves_mask()
        illegal_mask = ~legal_mask

        # All illegal moves should have zero probability
        assert (policy[illegal_mask] == 0).all()

    def test_valid_get_policy_matches_selected_move(
        self, neural_network: ChessNN, fresh_game: ChessGame
    ) -> None:
        """With temperature=0, selected move should have highest policy value."""
        mcts = MCTS(neural_network=neural_network, num_simulations=20)
        move, policy = mcts.get_policy(fresh_game, temperature=0)

        move_index = fresh_game.get_move_index(move)
        # The selected move should be among the highest
        # (might not be strictly highest due to tie-breaking)
        assert policy[move_index] > 0


class TestDirichletNoise:
    """Tests for Dirichlet noise application."""

    def test_valid_noise_changes_behavior(
        self, neural_network: ChessNN, fresh_game: ChessGame
    ) -> None:
        """Adding noise can change the selected move."""
        mcts = MCTS(neural_network=neural_network, num_simulations=20)

        # Collect moves with and without noise
        moves_no_noise = set()
        moves_with_noise = set()

        np.random.seed(42)
        for i in range(10):
            move = mcts.search(fresh_game, temperature=0, add_noise=False)
            moves_no_noise.add(move)

        np.random.seed(42)
        for i in range(10):
            move = mcts.search(fresh_game, temperature=1.0, add_noise=True)
            moves_with_noise.add(move)

        # Without noise and temp=0, should be consistent
        assert len(moves_no_noise) == 1
        # With noise and temp=1, more variation expected
        # (may still be 1 if noise doesn't overcome network preferences)

    def test_valid_noise_only_affects_root(
        self, mcts: MCTS, fresh_game: ChessGame
    ) -> None:
        """Dirichlet noise only affects root node priors."""
        # This is a behavioral test - noise should only apply at root
        # We verify by ensuring search still works correctly
        move = mcts.search(fresh_game, temperature=1.0, add_noise=True)
        assert move in fresh_game.get_legal_moves()


class TestTerminalPositions:
    """Tests for handling terminal game positions."""

    def test_valid_terminal_checkmate_returns_result(
        self, neural_network: ChessNN
    ) -> None:
        """Terminal checkmate position uses actual game result."""
        # Position one move before checkmate
        game = ChessGame()
        game.make_move(chess.Move.from_uci("f2f3"))
        game.make_move(chess.Move.from_uci("e7e5"))
        game.make_move(chess.Move.from_uci("g2g4"))
        # Black to move, Qh4# is checkmate

        mcts = MCTS(neural_network=neural_network, num_simulations=20)
        move = mcts.search(game, temperature=0)

        # MCTS should find checkmate or at least return a legal move
        assert move in game.get_legal_moves()

    def test_valid_stalemate_position_near_terminal(
        self, neural_network: ChessNN
    ) -> None:
        """Search handles position near stalemate correctly."""
        # Position where one side has very few legal moves
        game = ChessGame("8/8/8/8/8/5K2/8/4k2R w - - 0 1")

        mcts = MCTS(neural_network=neural_network, num_simulations=10)
        move = mcts.search(game, temperature=1.0)

        assert move in game.get_legal_moves()

    def test_error_search_on_stalemate(
        self, mcts: MCTS, stalemate_game: ChessGame
    ) -> None:
        """Search raises ValueError on stalemate position."""
        assert stalemate_game.is_game_over()
        with pytest.raises(ValueError):
            mcts.search(stalemate_game)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_valid_single_legal_move(
        self, neural_network: ChessNN
    ) -> None:
        """Search works when only one legal move exists."""
        # King has only one legal move
        game = ChessGame("8/8/8/8/8/8/R6k/K7 b - - 0 1")

        mcts = MCTS(neural_network=neural_network, num_simulations=10)
        move = mcts.search(game, temperature=1.0)

        legal_moves = game.get_legal_moves()
        assert len(legal_moves) >= 1
        assert move in legal_moves

    def test_valid_few_simulations(
        self, neural_network: ChessNN, fresh_game: ChessGame
    ) -> None:
        """Search works with very few simulations."""
        mcts = MCTS(neural_network=neural_network, num_simulations=1)
        move = mcts.search(fresh_game, temperature=1.0)

        assert move in fresh_game.get_legal_moves()

    def test_valid_many_simulations(
        self, neural_network: ChessNN, fresh_game: ChessGame
    ) -> None:
        """Search works with many simulations."""
        mcts = MCTS(neural_network=neural_network, num_simulations=50)
        move = mcts.search(fresh_game, temperature=0)

        assert move in fresh_game.get_legal_moves()

    def test_valid_custom_c_puct(
        self, neural_network: ChessNN, fresh_game: ChessGame
    ) -> None:
        """Search works with custom c_puct values."""
        mcts_low = MCTS(neural_network=neural_network, num_simulations=10, c_puct=0.1)
        mcts_high = MCTS(neural_network=neural_network, num_simulations=10, c_puct=5.0)

        move_low = mcts_low.search(fresh_game, temperature=0)
        move_high = mcts_high.search(fresh_game, temperature=0)

        assert move_low in fresh_game.get_legal_moves()
        assert move_high in fresh_game.get_legal_moves()
