"""
Tests for Data Preparation module.

Follows TEST_GUIDELINES.md conventions with class-based organization
and standardized naming prefixes.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import chess.pgn
import io

from src.data_preparation import (
    parse_eval,
    DataFilter,
    DataDownloader,
    DataExtractor,
    DatasetBuilder,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def data_filter() -> DataFilter:
    """Default DataFilter instance."""
    return DataFilter()


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_pgn_content() -> str:
    """Sample PGN content with evaluations."""
    return """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[Date "2024.01.15"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1950"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 { [%eval 0.3] } 1... e5 { [%eval 0.28] } 2. Nf3 { [%eval 0.35] } 2... Nc6 { [%eval 0.32] } 3. Bb5 { [%eval 0.4] } 3... a6 { [%eval 0.38] } 4. Ba4 { [%eval 0.35] } 4... Nf6 { [%eval 0.3] } 5. O-O { [%eval 0.32] } 5... Be7 { [%eval 0.28] } 6. Re1 { [%eval 0.3] } 6... b5 { [%eval 0.35] } 7. Bb3 { [%eval 0.32] } 7... O-O { [%eval 0.3] } 8. c3 { [%eval 0.28] } 8... d6 { [%eval 0.3] } 9. h3 { [%eval 0.25] } 9... Na5 { [%eval 0.28] } 10. Bc2 { [%eval 0.3] } 10... c5 { [%eval 0.32] } 1-0

"""


@pytest.fixture
def sample_pgn_file(temp_dir: str, sample_pgn_content: str) -> str:
    """Create a temporary PGN file."""
    pgn_path = os.path.join(temp_dir, "test.pgn")
    with open(pgn_path, "w") as f:
        f.write(sample_pgn_content)
    return pgn_path


@pytest.fixture
def sample_zst_file(temp_dir: str, sample_pgn_content: str) -> str:
    """Create a temporary compressed PGN file."""
    import zstandard as zstd

    zst_path = os.path.join(temp_dir, "test.pgn.zst")
    cctx = zstd.ZstdCompressor()
    with open(zst_path, "wb") as f:
        f.write(cctx.compress(sample_pgn_content.encode("utf-8")))
    return zst_path


# =============================================================================
# Test Classes
# =============================================================================


class TestParseEval:
    """Tests for parse_eval() function."""

    # -------------------------------------------------------------------------
    # Valid eval parsing tests
    # -------------------------------------------------------------------------

    def test_valid_positive_eval(self) -> None:
        """Parses positive evaluation correctly."""
        assert parse_eval("[%eval 0.35]") == 35

    def test_valid_negative_eval(self) -> None:
        """Parses negative evaluation correctly."""
        assert parse_eval("[%eval -1.5]") == -150

    def test_valid_zero_eval(self) -> None:
        """Parses zero evaluation."""
        assert parse_eval("[%eval 0.0]") == 0

    def test_valid_large_positive_eval(self) -> None:
        """Parses large positive evaluation."""
        assert parse_eval("[%eval 5.25]") == 525

    def test_valid_large_negative_eval(self) -> None:
        """Parses large negative evaluation."""
        assert parse_eval("[%eval -3.8]") == -380

    def test_valid_integer_eval(self) -> None:
        """Parses integer evaluation without decimal."""
        assert parse_eval("[%eval 2]") == 200

    def test_valid_eval_in_longer_comment(self) -> None:
        """Parses eval embedded in longer comment."""
        assert parse_eval("Some text [%eval 0.5] more text") == 50

    # -------------------------------------------------------------------------
    # Mate score tests
    # -------------------------------------------------------------------------

    def test_valid_mate_in_n_returns_none(self) -> None:
        """Mate in N returns None (excluded from balanced range)."""
        assert parse_eval("[%eval #4]") is None

    def test_valid_mated_in_n_returns_none(self) -> None:
        """Mated in N returns None."""
        assert parse_eval("[%eval #-4]") is None

    def test_valid_mate_in_1_returns_none(self) -> None:
        """Mate in 1 returns None."""
        assert parse_eval("[%eval #1]") is None

    # -------------------------------------------------------------------------
    # No eval tests
    # -------------------------------------------------------------------------

    def test_valid_no_eval_returns_none(self) -> None:
        """Returns None when no eval present."""
        assert parse_eval("Just a comment") is None

    def test_valid_empty_string_returns_none(self) -> None:
        """Returns None for empty string."""
        assert parse_eval("") is None

    def test_valid_malformed_eval_returns_none(self) -> None:
        """Returns None for malformed eval."""
        assert parse_eval("[%eval abc]") is None


class TestDataFilterInit:
    """Tests for DataFilter initialization."""

    def test_valid_default_eval_range(self) -> None:
        """Default eval range is (-200, 200)."""
        df = DataFilter()
        assert df.eval_range_cp == (-200, 200)

    def test_valid_default_min_ply(self) -> None:
        """Default min_ply is 16."""
        df = DataFilter()
        assert df.min_ply == 16

    def test_valid_default_min_game_length(self) -> None:
        """Default min_game_length is 20."""
        df = DataFilter()
        assert df.min_game_length == 20

    def test_valid_default_require_eval(self) -> None:
        """Default require_eval is True."""
        df = DataFilter()
        assert df.require_eval is True

    def test_valid_custom_eval_range(self) -> None:
        """Custom eval range is stored."""
        df = DataFilter(eval_range_cp=(-100, 100))
        assert df.eval_range_cp == (-100, 100)

    def test_valid_custom_min_ply(self) -> None:
        """Custom min_ply is stored."""
        df = DataFilter(min_ply=20)
        assert df.min_ply == 20


class TestDataFilterFilterPosition:
    """Tests for DataFilter.filter_position() method."""

    # -------------------------------------------------------------------------
    # Valid position tests
    # -------------------------------------------------------------------------

    def test_valid_balanced_position_passes(self, data_filter: DataFilter) -> None:
        """Balanced position within eval range passes."""
        assert data_filter.filter_position(eval_cp=50, ply=20, game_length=30) is True

    def test_valid_zero_eval_passes(self, data_filter: DataFilter) -> None:
        """Zero eval position passes."""
        assert data_filter.filter_position(eval_cp=0, ply=20, game_length=30) is True

    def test_valid_edge_positive_eval_passes(self, data_filter: DataFilter) -> None:
        """Position at positive eval boundary passes."""
        assert data_filter.filter_position(eval_cp=200, ply=20, game_length=30) is True

    def test_valid_edge_negative_eval_passes(self, data_filter: DataFilter) -> None:
        """Position at negative eval boundary passes."""
        assert data_filter.filter_position(eval_cp=-200, ply=20, game_length=30) is True

    def test_valid_edge_min_ply_passes(self, data_filter: DataFilter) -> None:
        """Position at minimum ply passes."""
        assert data_filter.filter_position(eval_cp=50, ply=16, game_length=30) is True

    # -------------------------------------------------------------------------
    # Rejected position tests
    # -------------------------------------------------------------------------

    def test_valid_large_positive_eval_rejected(self, data_filter: DataFilter) -> None:
        """Position with large positive eval is rejected."""
        assert data_filter.filter_position(eval_cp=300, ply=20, game_length=30) is False

    def test_valid_large_negative_eval_rejected(self, data_filter: DataFilter) -> None:
        """Position with large negative eval is rejected."""
        assert data_filter.filter_position(eval_cp=-300, ply=20, game_length=30) is False

    def test_valid_early_ply_rejected(self, data_filter: DataFilter) -> None:
        """Position in opening (early ply) is rejected."""
        assert data_filter.filter_position(eval_cp=50, ply=10, game_length=30) is False

    def test_valid_short_game_rejected(self, data_filter: DataFilter) -> None:
        """Position from short game is rejected."""
        assert data_filter.filter_position(eval_cp=50, ply=20, game_length=15) is False

    def test_valid_no_eval_rejected(self, data_filter: DataFilter) -> None:
        """Position without eval is rejected when require_eval is True."""
        assert data_filter.filter_position(eval_cp=None, ply=20, game_length=30) is False

    def test_valid_no_eval_passes_when_not_required(self) -> None:
        """Position without eval passes when require_eval is False."""
        df = DataFilter(require_eval=False)
        assert df.filter_position(eval_cp=None, ply=20, game_length=30) is True


class TestDataDownloaderInit:
    """Tests for DataDownloader initialization."""

    def test_valid_default_output_dir(self) -> None:
        """Default output_dir is data/raw."""
        dd = DataDownloader()
        assert dd.output_dir == "data/raw"

    def test_valid_custom_output_dir(self, temp_dir: str) -> None:
        """Custom output_dir is stored."""
        dd = DataDownloader(output_dir=temp_dir)
        assert dd.output_dir == temp_dir


class TestDataDownloaderGetDownloadUrl:
    """Tests for DataDownloader.get_download_url() method."""

    def test_valid_url_format(self) -> None:
        """URL is correctly formatted for Lichess database."""
        dd = DataDownloader()
        url = dd.get_download_url(2024, 1)
        assert "lichess.org" in url
        assert "2024-01" in url
        assert ".pgn.zst" in url

    def test_valid_url_with_different_months(self) -> None:
        """URL correctly handles different months."""
        dd = DataDownloader()
        url_jan = dd.get_download_url(2024, 1)
        url_dec = dd.get_download_url(2023, 12)

        assert "2024-01" in url_jan
        assert "2023-12" in url_dec


class TestDataDownloaderDownloadMonth:
    """Tests for DataDownloader.download_month() with mocking."""

    @patch("src.data_preparation.requests.get")
    def test_valid_download_creates_file(
        self, mock_get: Mock, temp_dir: str
    ) -> None:
        """Download creates file in output directory."""
        # Mock response with small content
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b"fake pgn data"])
        mock_response.headers = {"content-length": "13"}
        mock_response.raise_for_status = Mock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        dd = DataDownloader(output_dir=temp_dir)
        result_path = dd.download_month(2024, 1)

        assert os.path.exists(result_path)
        assert "2024-01" in result_path

    @patch("src.data_preparation.requests.get")
    def test_valid_download_returns_path(
        self, mock_get: Mock, temp_dir: str
    ) -> None:
        """Download returns the path to the downloaded file."""
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b"data"])
        mock_response.headers = {"content-length": "4"}
        mock_response.raise_for_status = Mock()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
        mock_get.return_value = mock_response

        dd = DataDownloader(output_dir=temp_dir)
        result = dd.download_month(2024, 1)

        assert isinstance(result, str)
        assert result.endswith(".pgn.zst")

    @patch("src.data_preparation.requests.get")
    def test_valid_skip_existing_file(
        self, mock_get: Mock, temp_dir: str
    ) -> None:
        """Download skips existing file when overwrite=False."""
        # Create existing file
        existing_path = os.path.join(
            temp_dir, "lichess_db_standard_rated_2024-01.pgn.zst"
        )
        with open(existing_path, "w") as f:
            f.write("existing")

        dd = DataDownloader(output_dir=temp_dir)
        result = dd.download_month(2024, 1, overwrite=False)

        # Should not have called requests.get
        mock_get.assert_not_called()
        assert result == existing_path


class TestDataExtractorInit:
    """Tests for DataExtractor initialization."""

    def test_valid_stores_filter(self, data_filter: DataFilter) -> None:
        """DataExtractor stores the provided filter."""
        extractor = DataExtractor(filter=data_filter)
        assert extractor.filter is data_filter


class TestDataExtractorExtractFromPgn:
    """Tests for DataExtractor.extract_from_pgn() method."""

    def test_valid_extracts_positions(
        self, data_filter: DataFilter, sample_pgn_file: str
    ) -> None:
        """Extracts positions from PGN file."""
        # Use permissive filter for testing
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        extractor = DataExtractor(filter=permissive_filter)

        positions = list(extractor.extract_from_pgn(sample_pgn_file, progress=False))

        assert len(positions) > 0

    def test_valid_position_has_required_fields(
        self, sample_pgn_file: str
    ) -> None:
        """Extracted positions have fen, move, and eval_cp fields."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        extractor = DataExtractor(filter=permissive_filter)

        positions = list(extractor.extract_from_pgn(sample_pgn_file, progress=False))

        for pos in positions:
            assert "fen" in pos
            assert "move" in pos
            assert "eval_cp" in pos

    def test_valid_fen_is_valid_position(
        self, sample_pgn_file: str
    ) -> None:
        """Extracted FEN strings are valid chess positions."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        extractor = DataExtractor(filter=permissive_filter)

        positions = list(extractor.extract_from_pgn(sample_pgn_file, progress=False))

        for pos in positions:
            board = chess.Board(pos["fen"])
            assert board.is_valid()

    def test_valid_respects_max_positions(
        self, sample_pgn_file: str
    ) -> None:
        """Respects max_positions limit."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        extractor = DataExtractor(filter=permissive_filter)

        positions = list(
            extractor.extract_from_pgn(sample_pgn_file, max_positions=3, progress=False)
        )

        assert len(positions) <= 3


class TestDataExtractorExtractFromZst:
    """Tests for DataExtractor.extract_from_zst() method."""

    def test_valid_extracts_from_compressed(
        self, sample_zst_file: str
    ) -> None:
        """Extracts positions from compressed PGN file."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        extractor = DataExtractor(filter=permissive_filter)

        positions = list(extractor.extract_from_zst(sample_zst_file, progress=False))

        assert len(positions) > 0

    def test_valid_zst_same_as_pgn(
        self, sample_pgn_file: str, sample_zst_file: str
    ) -> None:
        """Compressed and uncompressed extraction yield same results."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        extractor = DataExtractor(filter=permissive_filter)

        pgn_positions = list(extractor.extract_from_pgn(sample_pgn_file, progress=False))
        zst_positions = list(extractor.extract_from_zst(sample_zst_file, progress=False))

        assert len(pgn_positions) == len(zst_positions)
        for pgn_pos, zst_pos in zip(pgn_positions, zst_positions):
            assert pgn_pos["fen"] == zst_pos["fen"]
            assert pgn_pos["move"] == zst_pos["move"]


class TestDatasetBuilderInit:
    """Tests for DatasetBuilder initialization."""

    def test_valid_default_output_dir(self) -> None:
        """Default output_dir is data/processed."""
        builder = DatasetBuilder()
        assert builder.output_dir == "data/processed"

    def test_valid_custom_output_dir(self, temp_dir: str) -> None:
        """Custom output_dir is stored."""
        builder = DatasetBuilder(output_dir=temp_dir)
        assert builder.output_dir == temp_dir


class TestDatasetBuilderBuild:
    """Tests for DatasetBuilder.build() method."""

    def test_valid_creates_jsonl_file(
        self, temp_dir: str, sample_pgn_file: str
    ) -> None:
        """Build creates a JSONL file."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        builder = DatasetBuilder(output_dir=temp_dir)

        output_path = builder.build(
            source_paths=[sample_pgn_file],
            output_name="test_dataset",
            filter=permissive_filter,
            shuffle=False,
        )

        assert os.path.exists(output_path)
        assert output_path.endswith(".jsonl")

    def test_valid_jsonl_format(
        self, temp_dir: str, sample_pgn_file: str
    ) -> None:
        """Output file is valid JSONL format."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        builder = DatasetBuilder(output_dir=temp_dir)

        output_path = builder.build(
            source_paths=[sample_pgn_file],
            output_name="test_dataset",
            filter=permissive_filter,
            shuffle=False,
        )

        with open(output_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                assert "fen" in record
                assert "move" in record
                assert "eval_cp" in record

    def test_valid_respects_max_positions(
        self, temp_dir: str, sample_pgn_file: str
    ) -> None:
        """Build respects max_positions limit."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        builder = DatasetBuilder(output_dir=temp_dir)

        output_path = builder.build(
            source_paths=[sample_pgn_file],
            output_name="test_limited",
            filter=permissive_filter,
            max_positions=3,
            shuffle=False,
        )

        with open(output_path, "r") as f:
            lines = f.readlines()
        assert len(lines) <= 3

    def test_valid_creates_metadata_file(
        self, temp_dir: str, sample_pgn_file: str
    ) -> None:
        """Build creates metadata JSON file."""
        permissive_filter = DataFilter(min_ply=0, min_game_length=1)
        builder = DatasetBuilder(output_dir=temp_dir)

        output_path = builder.build(
            source_paths=[sample_pgn_file],
            output_name="test_meta",
            filter=permissive_filter,
            shuffle=False,
        )

        meta_path = output_path.replace(".jsonl", "_meta.json")
        assert os.path.exists(meta_path)

        with open(meta_path, "r") as f:
            meta = json.load(f)
            assert "num_positions" in meta
            assert "filters" in meta


class TestDatasetBuilderMerge:
    """Tests for DatasetBuilder.merge() method."""

    def test_valid_merges_datasets(self, temp_dir: str) -> None:
        """Merge combines multiple JSONL files."""
        # Create two small JSONL files
        file1 = os.path.join(temp_dir, "data1.jsonl")
        file2 = os.path.join(temp_dir, "data2.jsonl")

        with open(file1, "w") as f:
            f.write('{"fen": "fen1", "move": "e2e4", "eval_cp": 30}\n')
            f.write('{"fen": "fen2", "move": "d2d4", "eval_cp": 20}\n')

        with open(file2, "w") as f:
            f.write('{"fen": "fen3", "move": "c2c4", "eval_cp": 10}\n')

        builder = DatasetBuilder(output_dir=temp_dir)
        output_path = builder.merge(
            input_paths=[file1, file2],
            output_name="merged",
            shuffle=False,
        )

        with open(output_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 3

    def test_valid_merge_preserves_records(self, temp_dir: str) -> None:
        """Merge preserves all record data."""
        file1 = os.path.join(temp_dir, "data1.jsonl")
        with open(file1, "w") as f:
            f.write('{"fen": "test_fen", "move": "e2e4", "eval_cp": 42}\n')

        builder = DatasetBuilder(output_dir=temp_dir)
        output_path = builder.merge(
            input_paths=[file1],
            output_name="single_merge",
            shuffle=False,
        )

        with open(output_path, "r") as f:
            record = json.loads(f.readline())

        assert record["fen"] == "test_fen"
        assert record["move"] == "e2e4"
        assert record["eval_cp"] == 42
