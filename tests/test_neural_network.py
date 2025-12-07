"""
Comprehensive tests for ChessNN neural network component.

Test classes:
- TestChessNNInitialization
- TestChessNNModelArchitecture
- TestChessNNForwardPass
- TestChessNNIntegration
- TestChessNNResidualBlocks
- TestChessNNSaveLoad
- TestChessNNTraining
- TestChessNNEdgeCases
"""

import pytest
import numpy as np
import tensorflow as tf
import chess
from src.chess_game import ChessGame
from src.neural_network import ChessNN


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def chess_nn() -> ChessNN:
    """Fresh neural network instance with default parameters."""
    return ChessNN(num_residual_blocks=10, num_filters=128)


@pytest.fixture
def small_nn() -> ChessNN:
    """Smaller network for faster testing."""
    return ChessNN(num_residual_blocks=2, num_filters=64)


@pytest.fixture
def built_model(chess_nn: ChessNN) -> ChessNN:
    """Neural network with model built."""
    chess_nn.build_model()
    return chess_nn


@pytest.fixture
def built_small_model(small_nn: ChessNN) -> ChessNN:
    """Small neural network with model built."""
    small_nn.build_model()
    return small_nn


@pytest.fixture
def sample_board() -> np.ndarray:
    """Sample board tensor from starting position."""
    game = ChessGame()
    return game.get_canonical_board()


@pytest.fixture
def sample_batch() -> np.ndarray:
    """Batch of board tensors from different positions."""
    game = ChessGame()
    boards = []

    boards.append(game.get_canonical_board())
    game.make_move(chess.Move.from_uci("e2e4"))
    boards.append(game.get_canonical_board())
    game.make_move(chess.Move.from_uci("e7e5"))
    boards.append(game.get_canonical_board())

    return np.array(boards, dtype=np.float32)


@pytest.fixture
def endgame_board() -> np.ndarray:
    """Endgame position board tensor."""
    game = ChessGame(fen="7k/8/8/8/8/8/4Q3/4K3 w - - 0 1")
    return game.get_canonical_board()


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestChessNNInitialization:
    """Tests for ChessNN initialization and configuration."""

    # -------------------------------------------------------------------------
    # Valid initialization tests
    # -------------------------------------------------------------------------

    def test_valid_default_initialization(self) -> None:
        """ChessNN can be instantiated with default parameters."""
        nn = ChessNN()
        assert nn is not None

    @pytest.mark.parametrize('blocks,filters,lr', [
        pytest.param(5, 64, 0.01, id="small_network"),
        pytest.param(10, 128, 0.001, id="default_network"),
        pytest.param(20, 256, 0.0001, id="large_network"),
    ])
    def test_valid_custom_parameters(
        self,
        blocks: int,
        filters: int,
        lr: float
    ) -> None:
        """ChessNN accepts custom architecture parameters."""
        nn = ChessNN(num_residual_blocks=blocks, num_filters=filters, learning_rate=lr)
        assert nn.num_residual_blocks == blocks
        assert nn.num_filters == filters
        assert nn.learning_rate == lr

    def test_valid_build_model_succeeds(self, chess_nn: ChessNN) -> None:
        """build_model() completes without errors."""
        model = chess_nn.build_model()
        assert model is not None


class TestChessNNModelArchitecture:
    """Tests for neural network architecture."""

    # -------------------------------------------------------------------------
    # Valid architecture tests
    # -------------------------------------------------------------------------

    def test_valid_input_shape(self, built_model: ChessNN) -> None:
        """Model accepts (batch, 8, 8, 14) input."""
        input_shape = built_model.model.input_shape
        assert input_shape == (None, 8, 8, 14)

    def test_valid_output_shapes(self, built_model: ChessNN) -> None:
        """Model outputs have correct shapes."""
        outputs = built_model.model.output
        assert outputs['policy'].shape == (None, 4672)
        assert outputs['value'].shape == (None, 1)

    def test_valid_parameter_count(self, built_model: ChessNN) -> None:
        """Model has approximately 3-4M parameters."""
        total_params = built_model.model.count_params()
        assert 3_000_000 <= total_params <= 5_000_000

    def test_valid_has_residual_blocks(self, built_model: ChessNN) -> None:
        """Model contains residual blocks."""
        layer_names = [layer.name for layer in built_model.model.layers]
        residual_layer_count = sum(1 for name in layer_names if 'res_' in name)
        assert residual_layer_count > 20

    def test_valid_has_policy_head(self, built_model: ChessNN) -> None:
        """Model contains policy head layers."""
        layer_names = [layer.name for layer in built_model.model.layers]
        assert 'policy_output' in layer_names
        assert 'policy_dense' in layer_names

    def test_valid_has_value_head(self, built_model: ChessNN) -> None:
        """Model contains value head layers."""
        layer_names = [layer.name for layer in built_model.model.layers]
        assert 'value_output' in layer_names
        assert 'value_dense1' in layer_names
        assert 'value_dense2' in layer_names

    def test_valid_has_batch_normalization(self, built_model: ChessNN) -> None:
        """Model contains BatchNormalization layers."""
        bn_layers = [layer for layer in built_model.model.layers
                     if isinstance(layer, tf.keras.layers.BatchNormalization)]
        assert len(bn_layers) >= 20

    def test_valid_model_is_keras_model(self, built_model: ChessNN) -> None:
        """Built model is a Keras Model instance."""
        assert isinstance(built_model.model, tf.keras.Model)

    def test_valid_model_is_compiled(self, built_model: ChessNN) -> None:
        """Model is compiled with optimizer and loss functions."""
        assert built_model.model.optimizer is not None
        assert isinstance(built_model.model.optimizer, tf.keras.optimizers.Optimizer)

    def test_valid_optimizer_learning_rate(self) -> None:
        """Optimizer is configured with specified learning rate."""
        custom_lr = 0.01
        nn = ChessNN(learning_rate=custom_lr)
        nn.build_model()
        assert nn.model.optimizer.learning_rate.numpy() == custom_lr

    def test_valid_has_loss_functions(self, built_model: ChessNN) -> None:
        """Model has loss functions configured."""
        assert built_model.model.loss is not None


class TestChessNNForwardPass:
    """Tests for forward pass / predict functionality."""

    # -------------------------------------------------------------------------
    # Valid prediction tests
    # -------------------------------------------------------------------------

    def test_valid_single_board_output_shapes(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Predict returns policy (4672,) and value scalar for single board."""
        policy, value = built_small_model.predict(sample_board)
        assert policy.shape == (4672,)
        assert isinstance(value, (float, np.floating))

    def test_valid_batch_output_shapes(
        self,
        built_small_model: ChessNN,
        sample_batch: np.ndarray
    ) -> None:
        """Predict returns correct shapes for batch input."""
        policy, value = built_small_model.predict(sample_batch)
        batch_size = sample_batch.shape[0]
        assert policy.shape == (batch_size, 4672)
        assert value.shape == (batch_size, 1)

    def test_valid_policy_sums_to_one(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Policy output sums to 1.0 (valid probability distribution)."""
        policy, _ = built_small_model.predict(sample_board)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)

    def test_valid_policy_non_negative(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """All policy probabilities are non-negative."""
        policy, _ = built_small_model.predict(sample_board)
        assert np.all(policy >= 0.0)

    def test_valid_value_in_range(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Value output is in range [-1, 1]."""
        _, value = built_small_model.predict(sample_board)
        assert -1.0 <= value <= 1.0

    def test_valid_batch_policies_sum_to_one(
        self,
        built_small_model: ChessNN,
        sample_batch: np.ndarray
    ) -> None:
        """Each policy in batch sums to 1.0."""
        policy, _ = built_small_model.predict(sample_batch)
        for i in range(policy.shape[0]):
            assert np.isclose(policy[i].sum(), 1.0, atol=1e-5)

    def test_valid_batch_values_in_range(
        self,
        built_small_model: ChessNN,
        sample_batch: np.ndarray
    ) -> None:
        """All values in batch are in range [-1, 1]."""
        _, value = built_small_model.predict(sample_batch)
        assert np.all(value >= -1.0)
        assert np.all(value <= 1.0)

    def test_valid_output_dtype_float32(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Outputs have float32 dtype."""
        policy, value = built_small_model.predict(sample_board)
        assert policy.dtype == np.float32
        assert isinstance(value, (np.float32, float))

    def test_valid_handles_float64_input(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Network handles float64 input by converting to float32."""
        board_float64 = sample_board.astype(np.float64)
        policy, _ = built_small_model.predict(board_float64)
        assert policy.dtype == np.float32

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_error_predict_without_build(
        self,
        chess_nn: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Predict raises error if model not built."""
        with pytest.raises(ValueError, match="Model not built"):
            chess_nn.predict(sample_board)

    @pytest.mark.parametrize('invalid_shape,error_match', [
        pytest.param((8, 8, 12), "Invalid board shape", id="wrong_channels"),
        pytest.param((6, 6, 14), "Invalid board shape", id="wrong_dimensions"),
    ])
    def test_error_invalid_input_shape(
        self,
        built_small_model: ChessNN,
        invalid_shape: tuple,
        error_match: str
    ) -> None:
        """Predict rejects input with invalid shape."""
        invalid_board = np.zeros(invalid_shape, dtype=np.float32)
        with pytest.raises(ValueError, match=error_match):
            built_small_model.predict(invalid_board)

    def test_error_invalid_batch_shape(self, built_small_model: ChessNN) -> None:
        """Predict rejects batch with invalid board shape."""
        invalid_batch = np.zeros((2, 8, 8, 12), dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid board shape"):
            built_small_model.predict(invalid_batch)

    def test_error_wrong_dimensions(self, built_small_model: ChessNN) -> None:
        """Predict rejects input with wrong number of dimensions."""
        invalid_input = np.zeros((8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Invalid number of dimensions"):
            built_small_model.predict(invalid_input)


class TestChessNNIntegration:
    """Tests for integration with ChessGame."""

    # -------------------------------------------------------------------------
    # Valid integration tests
    # -------------------------------------------------------------------------

    def test_valid_accepts_chessgame_board(self, built_small_model: ChessNN) -> None:
        """Network accepts board from ChessGame.get_canonical_board()."""
        game = ChessGame()
        board = game.get_canonical_board()
        policy, value = built_small_model.predict(board)
        assert policy.shape == (4672,)

    def test_valid_starting_position(self, built_small_model: ChessNN) -> None:
        """Network produces valid output for starting position."""
        game = ChessGame()
        board = game.get_canonical_board()
        policy, value = built_small_model.predict(board)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value <= 1.0

    def test_valid_checkmate_position(self, built_small_model: ChessNN) -> None:
        """Network handles checkmate position."""
        game = ChessGame(fen="rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        board = game.get_canonical_board()
        policy, value = built_small_model.predict(board)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value <= 1.0

    def test_valid_endgame_position(
        self,
        built_small_model: ChessNN,
        endgame_board: np.ndarray
    ) -> None:
        """Network handles endgame position."""
        policy, value = built_small_model.predict(endgame_board)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value <= 1.0

    def test_valid_multiple_positions(self, built_small_model: ChessNN) -> None:
        """Network handles different positions from same game."""
        game = ChessGame()
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]

        for move_uci in moves:
            board = game.get_canonical_board()
            policy, value = built_small_model.predict(board)
            assert np.isclose(policy.sum(), 1.0, atol=1e-5)
            assert -1.0 <= value <= 1.0
            game.make_move(chess.Move.from_uci(move_uci))

    def test_valid_batch_from_chessgame(self, built_small_model: ChessNN) -> None:
        """Network handles batch of ChessGame boards."""
        game = ChessGame()
        boards = []

        for _ in range(5):
            boards.append(game.get_canonical_board())
            legal_moves = list(game.get_legal_moves())
            if legal_moves:
                game.make_move(legal_moves[0])

        batch = np.array(boards, dtype=np.float32)
        policy, value = built_small_model.predict(batch)
        assert policy.shape == (5, 4672)
        assert value.shape == (5, 1)


class TestChessNNResidualBlocks:
    """Tests for residual block implementation."""

    # -------------------------------------------------------------------------
    # Valid behavior tests
    # -------------------------------------------------------------------------

    def test_valid_has_skip_connections(self, built_model: ChessNN) -> None:
        """Residual blocks contain Add layers for skip connections."""
        add_layers = [layer for layer in built_model.model.layers
                      if isinstance(layer, tf.keras.layers.Add)]
        assert len(add_layers) >= built_model.num_residual_blocks

    def test_valid_preserves_spatial_shape(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Residual blocks maintain spatial dimensions."""
        layer_names = [layer.name for layer in built_small_model.model.layers]
        res_1_relu = None
        for layer in built_small_model.model.layers:
            if layer.name == 'res_1_relu2':
                res_1_relu = layer
                break

        if res_1_relu is not None:
            intermediate_model = tf.keras.Model(
                inputs=built_small_model.model.input,
                outputs=res_1_relu.output
            )

            board_batch = np.expand_dims(sample_board, axis=0)
            output = intermediate_model(board_batch, training=False)

            assert output.shape == (1, 8, 8, built_small_model.num_filters)

    def test_valid_uses_batch_normalization(self, built_model: ChessNN) -> None:
        """Residual blocks contain BatchNormalization layers."""
        layer_names = [layer.name for layer in built_model.model.layers]
        res_bn_count = sum(1 for name in layer_names if 'res_' in name and '_bn' in name)
        expected_bn_layers = built_model.num_residual_blocks * 2
        assert res_bn_count == expected_bn_layers


class TestChessNNSaveLoad:
    """Tests for save/load weights functionality."""

    # -------------------------------------------------------------------------
    # Valid save/load tests
    # -------------------------------------------------------------------------

    def test_valid_save_creates_file(
        self,
        built_small_model: ChessNN,
        tmp_path
    ) -> None:
        """save_weights creates a file."""
        filepath = tmp_path / "test_model.weights.h5"
        built_small_model.save_weights(str(filepath))
        assert filepath.exists()

    def test_valid_load_succeeds(
        self,
        built_small_model: ChessNN,
        tmp_path
    ) -> None:
        """load_weights completes without error."""
        filepath = tmp_path / "test_model.weights.h5"
        built_small_model.save_weights(str(filepath))

        new_nn = ChessNN(num_residual_blocks=2, num_filters=64)
        new_nn.build_model()
        new_nn.load_weights(str(filepath))

    def test_valid_save_load_preserves_predictions(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray,
        tmp_path
    ) -> None:
        """Saved and loaded model produces same predictions."""
        policy_before, value_before = built_small_model.predict(sample_board)

        filepath = tmp_path / "test_model.weights.h5"
        built_small_model.save_weights(str(filepath))

        new_nn = ChessNN(num_residual_blocks=2, num_filters=64)
        new_nn.build_model()
        new_nn.load_weights(str(filepath))

        policy_after, value_after = new_nn.predict(sample_board)

        np.testing.assert_array_almost_equal(policy_before, policy_after, decimal=5)
        np.testing.assert_almost_equal(value_before, value_after, decimal=5)

    # -------------------------------------------------------------------------
    # Error handling tests
    # -------------------------------------------------------------------------

    def test_error_load_without_build(self, chess_nn: ChessNN, tmp_path) -> None:
        """load_weights raises error if model not built."""
        filepath = tmp_path / "test_model.weights.h5"
        with pytest.raises(ValueError, match="Model not built"):
            chess_nn.load_weights(str(filepath))

    def test_error_save_without_build(self, chess_nn: ChessNN, tmp_path) -> None:
        """save_weights raises error if model not built."""
        filepath = tmp_path / "test_model.weights.h5"
        with pytest.raises(ValueError, match="Model not built"):
            chess_nn.save_weights(str(filepath))


class TestChessNNTraining:
    """Tests for training mode functionality."""

    # -------------------------------------------------------------------------
    # Valid training tests
    # -------------------------------------------------------------------------

    def test_valid_model_is_trainable(self, built_small_model: ChessNN) -> None:
        """Model has trainable parameters."""
        assert len(built_small_model.model.trainable_variables) > 0

    def test_valid_gradients_exist(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Gradients flow through all trainable parameters."""
        board_batch = np.expand_dims(sample_board, axis=0)

        policy_target = np.zeros((1, 4672), dtype=np.float32)
        policy_target[0, 0] = 1.0
        value_target = np.array([[0.5]], dtype=np.float32)

        with tf.GradientTape() as tape:
            outputs = built_small_model.model(board_batch, training=True)
            policy_loss = tf.keras.losses.categorical_crossentropy(
                policy_target, outputs['policy']
            )
            value_loss = tf.keras.losses.MeanSquaredError()(
                value_target, outputs['value']
            )
            total_loss = policy_loss + value_loss

        gradients = tape.gradient(total_loss, built_small_model.model.trainable_variables)

        for grad in gradients:
            assert grad is not None

    def test_valid_no_nan_or_inf(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Forward pass produces no NaN or Inf values."""
        policy, value = built_small_model.predict(sample_board)
        assert not np.any(np.isnan(policy))
        assert not np.any(np.isinf(policy))
        assert not np.isnan(value)
        assert not np.isinf(value)

    def test_valid_training_mode_works(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Model accepts both training=True and training=False."""
        board_batch = np.expand_dims(sample_board, axis=0)

        outputs_train = built_small_model.model(board_batch, training=True)
        policy_train = outputs_train['policy'].numpy()

        outputs_infer = built_small_model.model(board_batch, training=False)
        policy_infer = outputs_infer['policy'].numpy()

        assert policy_train.shape == policy_infer.shape

    def test_valid_can_train_with_fit(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Model can be trained using model.fit()."""
        board_batch = np.expand_dims(sample_board, axis=0)
        policy_target = np.zeros((1, 4672), dtype=np.float32)
        policy_target[0, 0] = 1.0
        value_target = np.array([[0.5]], dtype=np.float32)

        history = built_small_model.model.fit(
            board_batch,
            {'policy': policy_target, 'value': value_target},
            epochs=1,
            verbose=0
        )

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1

    def test_valid_can_train_on_batch(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Model can be trained using train_on_batch()."""
        board_batch = np.expand_dims(sample_board, axis=0)
        policy_target = np.zeros((1, 4672), dtype=np.float32)
        policy_target[0, 0] = 1.0
        value_target = np.array([[0.5]], dtype=np.float32)

        loss = built_small_model.model.train_on_batch(
            board_batch,
            {'policy': policy_target, 'value': value_target}
        )

        assert isinstance(loss, (float, list, np.ndarray))


class TestChessNNEdgeCases:
    """Tests for edge cases and boundary conditions."""

    # -------------------------------------------------------------------------
    # Edge case input tests
    # -------------------------------------------------------------------------

    def test_edge_zeros_input(self, built_small_model: ChessNN) -> None:
        """Network handles all-zeros input without errors."""
        zeros = np.zeros((8, 8, 14), dtype=np.float32)
        policy, value = built_small_model.predict(zeros)
        assert not np.any(np.isnan(policy))
        assert not np.isnan(value)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)

    def test_edge_ones_input(self, built_small_model: ChessNN) -> None:
        """Network handles all-ones input without errors."""
        ones = np.ones((8, 8, 14), dtype=np.float32)
        policy, value = built_small_model.predict(ones)
        assert not np.any(np.isnan(policy))
        assert not np.isnan(value)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)

    def test_edge_random_input(self, built_small_model: ChessNN) -> None:
        """Network handles random input without errors."""
        random_board = np.random.rand(8, 8, 14).astype(np.float32)
        policy, value = built_small_model.predict(random_board)
        assert not np.any(np.isnan(policy))
        assert not np.isnan(value)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)

    # -------------------------------------------------------------------------
    # Batch size edge cases
    # -------------------------------------------------------------------------

    def test_edge_batch_size_one(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Network handles batch size of 1."""
        batch = np.expand_dims(sample_board, axis=0)
        policy, value = built_small_model.predict(batch)
        assert policy.shape == (1, 4672)
        assert value.shape == (1, 1)

    def test_edge_large_batch(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Network handles large batch size (32)."""
        batch = np.stack([sample_board] * 32, axis=0)
        policy, value = built_small_model.predict(batch)
        assert policy.shape == (32, 4672)
        assert value.shape == (32, 1)

    # -------------------------------------------------------------------------
    # Determinism and consistency tests
    # -------------------------------------------------------------------------

    def test_edge_consistent_predictions(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray
    ) -> None:
        """Same input produces same output (deterministic inference)."""
        policy1, value1 = built_small_model.predict(sample_board)
        policy2, value2 = built_small_model.predict(sample_board)

        np.testing.assert_array_almost_equal(policy1, policy2, decimal=5)
        np.testing.assert_almost_equal(value1, value2, decimal=5)

    def test_edge_different_positions_different_outputs(
        self,
        built_small_model: ChessNN,
        sample_board: np.ndarray,
        endgame_board: np.ndarray
    ) -> None:
        """Different positions produce different outputs."""
        policy1, _ = built_small_model.predict(sample_board)
        policy2, _ = built_small_model.predict(endgame_board)

        assert not np.array_equal(policy1, policy2)

    # -------------------------------------------------------------------------
    # Architecture configuration tests
    # -------------------------------------------------------------------------

    def test_edge_small_architecture_fewer_params(self, small_nn: ChessNN) -> None:
        """Smaller architecture has fewer parameters."""
        small_nn.build_model()
        small_params = small_nn.model.count_params()

        large_nn = ChessNN(num_residual_blocks=10, num_filters=128)
        large_nn.build_model()
        large_params = large_nn.model.count_params()

        assert small_params < large_params

    def test_edge_variable_residual_blocks(self) -> None:
        """Number of residual blocks is configurable."""
        nn_5_blocks = ChessNN(num_residual_blocks=5, num_filters=64)
        nn_5_blocks.build_model()

        nn_3_blocks = ChessNN(num_residual_blocks=3, num_filters=64)
        nn_3_blocks.build_model()

        params_5 = nn_5_blocks.model.count_params()
        params_3 = nn_3_blocks.model.count_params()

        assert params_5 > params_3
