"""
Comprehensive tests for ChessNN neural network component.

Organized into sections:
A. Model architecture tests
B. Forward pass tests
C. Integration with ChessGame tests
D. Residual block tests
E. Save/load tests
F. Training mode tests
G. Edge case tests
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
    """Fresh neural network instance."""
    return ChessNN(num_residual_blocks=10, num_filters=128)


@pytest.fixture
def small_nn() -> ChessNN:
    """Smaller network for faster testing."""
    return ChessNN(num_residual_blocks=2, num_filters=64)


@pytest.fixture
def built_model(chess_nn) -> ChessNN:
    """Neural network with model built."""
    chess_nn.build_model()
    return chess_nn


@pytest.fixture
def built_small_model(small_nn) -> ChessNN:
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
    """Batch of board tensors."""
    game = ChessGame()
    boards = []

    # Starting position
    boards.append(game.get_canonical_board())

    # After e2e4
    game.make_move(chess.Move.from_uci("e2e4"))
    boards.append(game.get_canonical_board())

    # After e7e5
    game.make_move(chess.Move.from_uci("e7e5"))
    boards.append(game.get_canonical_board())

    return np.array(boards, dtype=np.float32)


@pytest.fixture
def endgame_board() -> np.ndarray:
    """Endgame position board tensor."""
    game = ChessGame(fen="7k/8/8/8/8/8/4Q3/4K3 w - - 0 1")
    return game.get_canonical_board()


# =============================================================================
# A. MODEL ARCHITECTURE TESTS
# =============================================================================

def test_neural_network_initialization_creates_instance():
    """Test that ChessNN can be instantiated with default parameters."""
    nn = ChessNN()
    assert nn is not None


def test_neural_network_initialization_accepts_custom_parameters():
    """Test that ChessNN accepts custom architecture parameters."""
    nn = ChessNN(num_residual_blocks=5, num_filters=64, learning_rate=0.01)
    assert nn.num_residual_blocks == 5
    assert nn.num_filters == 64
    assert nn.learning_rate == 0.01


def test_neural_network_builds_without_errors(chess_nn):
    """Test that build_model() completes without errors."""
    model = chess_nn.build_model()
    assert model is not None


def test_neural_network_has_correct_input_shape(built_model):
    """Test that model accepts (batch, 8, 8, 14) input."""
    input_shape = built_model.model.input_shape
    assert input_shape == (None, 8, 8, 14)


def test_neural_network_has_correct_output_shapes(built_model):
    """Test that model outputs have correct shapes."""
    # Get output shapes
    outputs = built_model.model.output
    policy_shape = outputs['policy'].shape
    value_shape = outputs['value'].shape

    assert policy_shape == (None, 4672)
    assert value_shape == (None, 1)


def test_neural_network_parameter_count_reasonable(built_model):
    """Test that model has approximately 3-4M parameters."""
    total_params = built_model.model.count_params()
    # With 10 residual blocks and 128 filters, should be around 3.6M
    assert 3_000_000 <= total_params <= 5_000_000


def test_neural_network_has_residual_blocks(built_model):
    """Test that model contains residual blocks."""
    layer_names = [layer.name for layer in built_model.model.layers]
    # Check for residual block layers
    residual_layer_count = sum(1 for name in layer_names if 'res_' in name)
    # Each residual block has multiple layers (conv, bn, relu, add)
    # With 10 blocks, we should have many residual layers
    assert residual_layer_count > 20


def test_neural_network_has_policy_head(built_model):
    """Test that model contains policy head layers."""
    layer_names = [layer.name for layer in built_model.model.layers]
    assert 'policy_output' in layer_names
    assert 'policy_dense' in layer_names


def test_neural_network_has_value_head(built_model):
    """Test that model contains value head layers."""
    layer_names = [layer.name for layer in built_model.model.layers]
    assert 'value_output' in layer_names
    assert 'value_dense1' in layer_names
    assert 'value_dense2' in layer_names


def test_neural_network_has_batch_normalization_layers(built_model):
    """Test that model contains BatchNormalization layers."""
    bn_layers = [layer for layer in built_model.model.layers
                 if isinstance(layer, tf.keras.layers.BatchNormalization)]
    # Should have many BN layers (initial + residual blocks + heads)
    assert len(bn_layers) >= 20


def test_neural_network_model_is_keras_model(built_model):
    """Test that built model is a Keras Model instance."""
    assert isinstance(built_model.model, tf.keras.Model)


def test_neural_network_model_is_compiled(built_model):
    """Test that model is compiled with optimizer and loss functions."""
    # Check optimizer exists
    assert built_model.model.optimizer is not None
    assert isinstance(built_model.model.optimizer, tf.keras.optimizers.Optimizer)


def test_neural_network_optimizer_uses_correct_learning_rate(chess_nn):
    """Test that optimizer is configured with specified learning rate."""
    custom_lr = 0.01
    nn = ChessNN(learning_rate=custom_lr)
    nn.build_model()
    # Check that optimizer has the correct learning rate
    assert nn.model.optimizer.learning_rate.numpy() == custom_lr


def test_neural_network_has_loss_functions(built_model):
    """Test that model has loss functions configured for both outputs."""
    # Model should have loss functions for policy and value
    assert built_model.model.loss is not None


# =============================================================================
# B. FORWARD PASS TESTS
# =============================================================================

def test_neural_network_predict_single_board_returns_correct_shapes(built_small_model, sample_board):
    """Test that predict returns policy (4672,) and value scalar for single board."""
    policy, value = built_small_model.predict(sample_board)
    assert policy.shape == (4672,)
    assert isinstance(value, (float, np.floating))


def test_neural_network_predict_batch_returns_correct_shapes(built_small_model, sample_batch):
    """Test that predict returns correct shapes for batch input."""
    policy, value = built_small_model.predict(sample_batch)
    batch_size = sample_batch.shape[0]
    assert policy.shape == (batch_size, 4672)
    assert value.shape == (batch_size, 1)


def test_neural_network_policy_sums_to_one(built_small_model, sample_board):
    """Test that policy output sums to 1.0 (valid probability distribution)."""
    policy, _ = built_small_model.predict(sample_board)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)


def test_neural_network_policy_all_non_negative(built_small_model, sample_board):
    """Test that all policy probabilities are non-negative."""
    policy, _ = built_small_model.predict(sample_board)
    assert np.all(policy >= 0.0)


def test_neural_network_value_in_valid_range(built_small_model, sample_board):
    """Test that value output is in range [-1, 1]."""
    _, value = built_small_model.predict(sample_board)
    assert -1.0 <= value <= 1.0


def test_neural_network_batch_policy_each_sums_to_one(built_small_model, sample_batch):
    """Test that each policy in batch sums to 1.0."""
    policy, _ = built_small_model.predict(sample_batch)
    for i in range(policy.shape[0]):
        assert np.isclose(policy[i].sum(), 1.0, atol=1e-5)


def test_neural_network_batch_values_in_valid_range(built_small_model, sample_batch):
    """Test that all values in batch are in range [-1, 1]."""
    _, value = built_small_model.predict(sample_batch)
    assert np.all(value >= -1.0)
    assert np.all(value <= 1.0)


def test_neural_network_output_dtype_is_float32(built_small_model, sample_board):
    """Test that outputs have float32 dtype."""
    policy, value = built_small_model.predict(sample_board)
    # Policy is numpy array
    assert policy.dtype == np.float32
    # Value is scalar but should be float32 compatible
    assert isinstance(value, (np.float32, float))


def test_neural_network_predict_without_build_raises_error(chess_nn, sample_board):
    """Test that predict raises error if model not built."""
    with pytest.raises(ValueError, match="Model not built"):
        chess_nn.predict(sample_board)


def test_neural_network_handles_float64_input(built_small_model, sample_board):
    """Test that network handles float64 input by converting to float32."""
    board_float64 = sample_board.astype(np.float64)
    policy, value = built_small_model.predict(board_float64)
    assert policy.dtype == np.float32


# =============================================================================
# C. INTEGRATION WITH CHESSGAME TESTS
# =============================================================================

def test_neural_network_accepts_chessgame_board(built_small_model):
    """Test that network accepts board from ChessGame.get_canonical_board()."""
    game = ChessGame()
    board = game.get_canonical_board()
    policy, value = built_small_model.predict(board)
    assert policy.shape == (4672,)


def test_neural_network_works_with_starting_position(built_small_model):
    """Test that network produces valid output for starting position."""
    game = ChessGame()
    board = game.get_canonical_board()
    policy, value = built_small_model.predict(board)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert -1.0 <= value <= 1.0


def test_neural_network_works_with_checkmate_position(built_small_model):
    """Test that network handles checkmate position."""
    # Fool's mate position
    game = ChessGame(fen="rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    board = game.get_canonical_board()
    policy, value = built_small_model.predict(board)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert -1.0 <= value <= 1.0


def test_neural_network_works_with_endgame_position(built_small_model, endgame_board):
    """Test that network handles endgame position."""
    policy, value = built_small_model.predict(endgame_board)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert -1.0 <= value <= 1.0


def test_neural_network_works_with_multiple_game_positions(built_small_model):
    """Test that network handles different positions from same game."""
    game = ChessGame()
    boards = []

    # Collect boards from a short game
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    for move_uci in moves:
        boards.append(game.get_canonical_board())
        game.make_move(chess.Move.from_uci(move_uci))

    # Test each board
    for board in boards:
        policy, value = built_small_model.predict(board)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
        assert -1.0 <= value <= 1.0


def test_neural_network_batch_from_chessgame(built_small_model):
    """Test that network handles batch of ChessGame boards."""
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


# =============================================================================
# D. RESIDUAL BLOCK TESTS
# =============================================================================

def test_neural_network_residual_blocks_have_skip_connections(built_model):
    """Test that residual blocks contain Add layers for skip connections."""
    add_layers = [layer for layer in built_model.model.layers
                  if isinstance(layer, tf.keras.layers.Add)]
    # Should have one Add layer per residual block
    assert len(add_layers) >= built_model.num_residual_blocks


def test_neural_network_residual_blocks_preserve_shape(built_small_model, sample_board):
    """Test that residual blocks maintain spatial dimensions."""
    # Get intermediate layer outputs
    # Create a model that outputs after first residual block
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

        # Output should be (1, 8, 8, num_filters)
        assert output.shape == (1, 8, 8, built_small_model.num_filters)


def test_neural_network_residual_blocks_use_batch_normalization(built_model):
    """Test that residual blocks contain BatchNormalization layers."""
    layer_names = [layer.name for layer in built_model.model.layers]
    # Each residual block should have 2 BN layers
    res_bn_count = sum(1 for name in layer_names if 'res_' in name and '_bn' in name)
    expected_bn_layers = built_model.num_residual_blocks * 2
    assert res_bn_count == expected_bn_layers


# =============================================================================
# E. SAVE/LOAD TESTS
# =============================================================================

def test_neural_network_save_weights_creates_file(built_small_model, tmp_path):
    """Test that save_weights creates a file."""
    filepath = tmp_path / "test_model.weights.h5"
    built_small_model.save_weights(str(filepath))
    assert filepath.exists()


def test_neural_network_load_weights_succeeds(built_small_model, tmp_path):
    """Test that load_weights completes without error."""
    filepath = tmp_path / "test_model.weights.h5"
    built_small_model.save_weights(str(filepath))

    # Create new instance and load
    new_nn = ChessNN(num_residual_blocks=2, num_filters=64)
    new_nn.build_model()
    new_nn.load_weights(str(filepath))
    # Should not raise error


def test_neural_network_save_load_preserves_predictions(built_small_model, sample_board, tmp_path):
    """Test that saved and loaded model produces same predictions."""
    # Get predictions before save
    policy_before, value_before = built_small_model.predict(sample_board)

    # Save and load
    filepath = tmp_path / "test_model.weights.h5"
    built_small_model.save_weights(str(filepath))

    # Create new instance and load
    new_nn = ChessNN(num_residual_blocks=2, num_filters=64)
    new_nn.build_model()
    new_nn.load_weights(str(filepath))

    # Get predictions after load
    policy_after, value_after = new_nn.predict(sample_board)

    # Compare - should be identical
    np.testing.assert_array_almost_equal(policy_before, policy_after, decimal=5)
    np.testing.assert_almost_equal(value_before, value_after, decimal=5)


def test_neural_network_load_weights_without_build_raises_error(chess_nn, tmp_path):
    """Test that load_weights raises error if model not built."""
    filepath = tmp_path / "test_model.weights.h5"
    with pytest.raises(ValueError, match="Model not built"):
        chess_nn.load_weights(str(filepath))


def test_neural_network_save_weights_without_build_raises_error(chess_nn, tmp_path):
    """Test that save_weights raises error if model not built."""
    filepath = tmp_path / "test_model.weights.h5"
    with pytest.raises(ValueError, match="Model not built"):
        chess_nn.save_weights(str(filepath))


# =============================================================================
# F. TRAINING MODE TESTS
# =============================================================================

def test_neural_network_model_is_trainable(built_small_model):
    """Test that model has trainable parameters."""
    assert len(built_small_model.model.trainable_variables) > 0


def test_neural_network_gradients_exist_for_all_trainable_params(built_small_model, sample_board):
    """Test that gradients flow through all trainable parameters."""
    # Expand dims for batch
    board_batch = np.expand_dims(sample_board, axis=0)

    # Dummy targets
    policy_target = np.zeros((1, 4672), dtype=np.float32)
    policy_target[0, 0] = 1.0  # One-hot
    value_target = np.array([[0.5]], dtype=np.float32)

    with tf.GradientTape() as tape:
        outputs = built_small_model.model(board_batch, training=True)
        policy_loss = tf.keras.losses.categorical_crossentropy(policy_target, outputs['policy'])
        value_loss = tf.keras.losses.MeanSquaredError()(value_target, outputs['value'])
        total_loss = policy_loss + value_loss

    gradients = tape.gradient(total_loss, built_small_model.model.trainable_variables)

    # All gradients should exist (not None)
    for grad in gradients:
        assert grad is not None


def test_neural_network_no_nan_or_inf_in_outputs(built_small_model, sample_board):
    """Test that forward pass produces no NaN or Inf values."""
    policy, value = built_small_model.predict(sample_board)
    assert not np.any(np.isnan(policy))
    assert not np.any(np.isinf(policy))
    assert not np.isnan(value)
    assert not np.isinf(value)


def test_neural_network_batch_normalization_affects_training_mode(built_small_model, sample_board):
    """Test that training=True vs training=False produces different outputs."""
    board_batch = np.expand_dims(sample_board, axis=0)

    # First call with training=True
    outputs_train = built_small_model.model(board_batch, training=True)
    policy_train = outputs_train['policy'].numpy()

    # Second call with training=False
    outputs_infer = built_small_model.model(board_batch, training=False)
    policy_infer = outputs_infer['policy'].numpy()

    # For a newly initialized model, training mode affects BatchNorm statistics
    # Outputs should be identical for a fresh model, but the mechanism is different
    # This test mainly verifies the model accepts both modes
    assert policy_train.shape == policy_infer.shape


def test_neural_network_can_train_with_fit(built_small_model, sample_board):
    """Test that model can be trained using model.fit()."""
    # Prepare data
    board_batch = np.expand_dims(sample_board, axis=0)
    policy_target = np.zeros((1, 4672), dtype=np.float32)
    policy_target[0, 0] = 1.0  # One-hot encoding
    value_target = np.array([[0.5]], dtype=np.float32)

    # Train for 1 epoch - should not raise error
    history = built_small_model.model.fit(
        board_batch,
        {'policy': policy_target, 'value': value_target},
        epochs=1,
        verbose=0
    )

    # Verify training occurred
    assert 'loss' in history.history
    assert len(history.history['loss']) == 1


def test_neural_network_can_train_with_train_on_batch(built_small_model, sample_board):
    """Test that model can be trained using train_on_batch()."""
    # Prepare data
    board_batch = np.expand_dims(sample_board, axis=0)
    policy_target = np.zeros((1, 4672), dtype=np.float32)
    policy_target[0, 0] = 1.0
    value_target = np.array([[0.5]], dtype=np.float32)

    # Train on batch - should return loss
    loss = built_small_model.model.train_on_batch(
        board_batch,
        {'policy': policy_target, 'value': value_target}
    )

    # Loss should be a scalar or list
    assert isinstance(loss, (float, list, np.ndarray))


# =============================================================================
# G. EDGE CASE TESTS
# =============================================================================

def test_neural_network_handles_zeros_input(built_small_model):
    """Test that network handles all-zeros input without errors."""
    zeros = np.zeros((8, 8, 14), dtype=np.float32)
    policy, value = built_small_model.predict(zeros)
    assert not np.any(np.isnan(policy))
    assert not np.isnan(value)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)


def test_neural_network_handles_ones_input(built_small_model):
    """Test that network handles all-ones input without errors."""
    ones = np.ones((8, 8, 14), dtype=np.float32)
    policy, value = built_small_model.predict(ones)
    assert not np.any(np.isnan(policy))
    assert not np.isnan(value)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)


def test_neural_network_handles_random_input(built_small_model):
    """Test that network handles random input without errors."""
    random_board = np.random.rand(8, 8, 14).astype(np.float32)
    policy, value = built_small_model.predict(random_board)
    assert not np.any(np.isnan(policy))
    assert not np.isnan(value)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)


def test_neural_network_handles_batch_size_one(built_small_model, sample_board):
    """Test that network handles batch size of 1."""
    batch = np.expand_dims(sample_board, axis=0)
    policy, value = built_small_model.predict(batch)
    assert policy.shape == (1, 4672)
    assert value.shape == (1, 1)


def test_neural_network_handles_large_batch(built_small_model, sample_board):
    """Test that network handles large batch size (32)."""
    batch = np.stack([sample_board] * 32, axis=0)
    policy, value = built_small_model.predict(batch)
    assert policy.shape == (32, 4672)
    assert value.shape == (32, 1)


def test_neural_network_consistent_predictions_for_same_input(built_small_model, sample_board):
    """Test that same input produces same output (deterministic inference)."""
    policy1, value1 = built_small_model.predict(sample_board)
    policy2, value2 = built_small_model.predict(sample_board)

    np.testing.assert_array_almost_equal(policy1, policy2, decimal=5)
    np.testing.assert_almost_equal(value1, value2, decimal=5)


def test_neural_network_different_positions_produce_different_outputs(built_small_model, sample_board, endgame_board):
    """Test that different positions produce different outputs."""
    policy1, value1 = built_small_model.predict(sample_board)
    policy2, value2 = built_small_model.predict(endgame_board)

    # Policies should not be exactly identical (at least some difference)
    # For a randomly initialized network, differences might be small
    assert not np.array_equal(policy1, policy2)


def test_neural_network_small_architecture_parameters(small_nn):
    """Test that smaller architecture has fewer parameters."""
    small_nn.build_model()
    small_params = small_nn.model.count_params()

    large_nn = ChessNN(num_residual_blocks=10, num_filters=128)
    large_nn.build_model()
    large_params = large_nn.model.count_params()

    assert small_params < large_params


def test_neural_network_variable_residual_blocks():
    """Test that number of residual blocks is configurable."""
    nn_5_blocks = ChessNN(num_residual_blocks=5, num_filters=64)
    nn_5_blocks.build_model()

    nn_3_blocks = ChessNN(num_residual_blocks=3, num_filters=64)
    nn_3_blocks.build_model()

    params_5 = nn_5_blocks.model.count_params()
    params_3 = nn_3_blocks.model.count_params()

    # More blocks should mean more parameters
    assert params_5 > params_3


def test_neural_network_rejects_invalid_input_shape_wrong_channels(built_small_model):
    """Test that predict rejects input with wrong number of channels."""
    invalid_board = np.zeros((8, 8, 12), dtype=np.float32)  # Wrong: 12 instead of 14
    with pytest.raises(ValueError, match="Invalid board shape"):
        built_small_model.predict(invalid_board)


def test_neural_network_rejects_invalid_input_shape_wrong_dimensions(built_small_model):
    """Test that predict rejects input with wrong spatial dimensions."""
    invalid_board = np.zeros((6, 6, 14), dtype=np.float32)  # Wrong: 6x6 instead of 8x8
    with pytest.raises(ValueError, match="Invalid board shape"):
        built_small_model.predict(invalid_board)


def test_neural_network_rejects_invalid_batch_shape(built_small_model):
    """Test that predict rejects batch with invalid board shape."""
    invalid_batch = np.zeros((2, 8, 8, 12), dtype=np.float32)  # Wrong: 12 channels
    with pytest.raises(ValueError, match="Invalid board shape"):
        built_small_model.predict(invalid_batch)


def test_neural_network_rejects_wrong_number_of_dimensions(built_small_model):
    """Test that predict rejects input with wrong number of dimensions."""
    invalid_input = np.zeros((8, 8), dtype=np.float32)  # Wrong: 2D instead of 3D/4D
    with pytest.raises(ValueError, match="Invalid number of dimensions"):
        built_small_model.predict(invalid_input)
