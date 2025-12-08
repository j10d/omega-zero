"""
Data Preparation module for OmegaZero.

Handles downloading, filtering, and processing chess game data from Lichess
for training. Creates JSONL datasets with positions, moves, and evaluations.
"""

import io
import json
import os
import random
import re
from collections.abc import Iterator
from datetime import datetime

import chess
import chess.pgn
import requests
import zstandard as zstd
from tqdm import tqdm


def parse_eval(comment: str) -> int | None:
    """
    Parse [%eval X] from PGN comment.

    Args:
        comment: PGN comment string that may contain evaluation.

    Returns:
        Centipawns (int), or None if no eval or mate.
        Positive = White advantage, Negative = Black advantage.
    """
    match = re.search(r'\[%eval (#?-?\d+\.?\d*)\]', comment)
    if not match:
        return None

    eval_str = match.group(1)

    # Handle mate scores - exclude these (outside balanced range)
    if eval_str.startswith('#'):
        return None

    try:
        # Convert to centipawns (eval is in pawns with decimal)
        return int(float(eval_str) * 100)
    except ValueError:
        return None


class DataFilter:
    """
    Filters positions based on evaluation criteria.

    Filters for balanced positions from well-played games, excluding
    opening moves and positions with large evaluation imbalances.
    """

    def __init__(
        self,
        eval_range_cp: tuple[int, int] = (-200, 200),
        min_ply: int = 16,
        min_game_length: int = 20,
        require_eval: bool = True,
    ):
        """
        Initialize filter with criteria.

        Args:
            eval_range_cp: Acceptable evaluation range in centipawns (default -200 to 200).
            min_ply: Minimum ply (half-moves) to skip opening (default 16 = 8 full moves).
            min_game_length: Minimum game length in moves (default 20).
            require_eval: Whether evaluation is required (default True).
        """
        self.eval_range_cp = eval_range_cp
        self.min_ply = min_ply
        self.min_game_length = min_game_length
        self.require_eval = require_eval

    def filter_position(
        self,
        eval_cp: int | None,
        ply: int,
        game_length: int,
    ) -> bool:
        """
        Check if a position passes all filters.

        Args:
            eval_cp: Position evaluation in centipawns, or None if not available.
            ply: Current ply (half-move number).
            game_length: Total game length in moves.

        Returns:
            True if position passes all filters, False otherwise.
        """
        # Check if eval is required but missing
        if self.require_eval and eval_cp is None:
            return False

        # Check eval range (skip if no eval and not required)
        if eval_cp is not None:
            if eval_cp < self.eval_range_cp[0] or eval_cp > self.eval_range_cp[1]:
                return False

        # Check minimum ply (skip opening)
        if ply < self.min_ply:
            return False

        # Check minimum game length
        if game_length < self.min_game_length:
            return False

        return True

    def filter_game(self, game: chess.pgn.Game) -> bool:
        """
        Check if a game is worth processing.

        Args:
            game: Parsed PGN game.

        Returns:
            True if game has evaluations and is long enough.
        """
        # Check for evaluations in first few moves
        node = game
        has_eval = False
        moves_count = 0

        while node.variations:
            node = node.variations[0]
            moves_count += 1
            if node.comment and "[%eval" in node.comment:
                has_eval = True
                break
            if moves_count > 5:
                break

        if self.require_eval and not has_eval:
            return False

        # Check game length
        total_moves = 0
        node = game
        while node.variations:
            node = node.variations[0]
            total_moves += 1

        if total_moves < self.min_game_length * 2:  # Convert to ply
            return False

        return True


class DataDownloader:
    """
    Downloads Lichess database files.

    Handles downloading monthly PGN archives from the Lichess database.
    """

    BASE_URL = "https://database.lichess.org/standard"

    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize downloader.

        Args:
            output_dir: Directory to save downloaded files.
        """
        self.output_dir = output_dir

    def get_download_url(self, year: int, month: int) -> str:
        """
        Construct download URL for a specific month.

        Args:
            year: Year (e.g., 2024).
            month: Month (1-12).

        Returns:
            URL for the compressed PGN file.
        """
        return f"{self.BASE_URL}/lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"

    def download_month(
        self,
        year: int,
        month: int,
        overwrite: bool = False,
    ) -> str:
        """
        Download a specific month's database.

        Args:
            year: Year (e.g., 2024).
            month: Month (1-12).
            overwrite: Whether to overwrite existing files.

        Returns:
            Path to the downloaded file.
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        filename = f"lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"
        output_path = os.path.join(self.output_dir, filename)

        # Skip if file exists and not overwriting
        if os.path.exists(output_path) and not overwrite:
            return output_path

        url = self.get_download_url(year, month)

        # Download with progress bar
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        return output_path

    def list_available_months(self) -> list[tuple[int, int]]:
        """
        List available months in the Lichess database.

        Returns:
            List of (year, month) tuples for available databases.
        """
        # Lichess database starts from 2013-01
        # Return a reasonable range - actual availability should be checked
        current_year = datetime.now().year
        current_month = datetime.now().month

        available = []
        for year in range(2013, current_year + 1):
            for month in range(1, 13):
                if year == current_year and month >= current_month:
                    break
                available.append((year, month))

        return available


class DataExtractor:
    """
    Extracts positions from filtered games.

    Handles both compressed (.pgn.zst) and uncompressed (.pgn) files
    with streaming to handle large files efficiently.
    """

    def __init__(self, filter: DataFilter):
        """
        Initialize extractor.

        Args:
            filter: DataFilter instance for position filtering.
        """
        self.filter = filter

    def extract_from_pgn(
        self,
        pgn_path: str,
        max_positions: int | None = None,
        progress: bool = True,
    ) -> Iterator[dict]:
        """
        Yield position records from a PGN file.

        Args:
            pgn_path: Path to PGN file.
            max_positions: Maximum positions to extract (None for all).
            progress: Whether to show progress bar.

        Yields:
            Position records: {"fen": str, "move": str, "eval_cp": int}
        """
        with open(pgn_path, "r") as f:
            yield from self._extract_from_stream(f, max_positions, progress)

    def extract_from_zst(
        self,
        zst_path: str,
        max_positions: int | None = None,
        progress: bool = True,
    ) -> Iterator[dict]:
        """
        Extract directly from compressed file (streaming).

        Args:
            zst_path: Path to compressed PGN file.
            max_positions: Maximum positions to extract.
            progress: Whether to show progress bar.

        Yields:
            Position records: {"fen": str, "move": str, "eval_cp": int}
        """
        dctx = zstd.ZstdDecompressor()
        with open(zst_path, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                yield from self._extract_from_stream(text_stream, max_positions, progress)

    def _extract_from_stream(
        self,
        stream,
        max_positions: int | None,
        progress: bool,
    ) -> Iterator[dict]:
        """
        Extract positions from a text stream.

        Args:
            stream: Text stream of PGN data.
            max_positions: Maximum positions to extract.
            progress: Whether to show progress bar.

        Yields:
            Position records.
        """
        position_count = 0
        game_count = 0

        pbar = tqdm(desc="Extracting positions", disable=not progress)

        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break

            game_count += 1

            # Quick filter for game quality
            if not self.filter.filter_game(game):
                continue

            # Count total moves for game length
            game_length = 0
            node = game
            while node.variations:
                node = node.variations[0]
                game_length += 1
            game_length = game_length // 2  # Convert ply to moves

            # Extract positions
            board = game.board()
            node = game
            ply = 0

            while node.variations:
                node = node.variations[0]
                move = node.move

                # Get evaluation from comment
                eval_cp = parse_eval(node.comment) if node.comment else None

                # Check filter
                if self.filter.filter_position(eval_cp, ply, game_length):
                    # Get FEN before move
                    fen = board.fen()
                    move_uci = move.uci()

                    yield {
                        "fen": fen,
                        "move": move_uci,
                        "eval_cp": eval_cp,
                    }

                    position_count += 1
                    pbar.update(1)

                    if max_positions and position_count >= max_positions:
                        pbar.close()
                        return

                # Make move
                board.push(move)
                ply += 1

        pbar.close()


class DatasetBuilder:
    """
    Builds the final processed dataset.

    Creates JSONL datasets from PGN/ZST files with optional shuffling.
    """

    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize builder.

        Args:
            output_dir: Directory to save processed datasets.
        """
        self.output_dir = output_dir

    def build(
        self,
        source_paths: list[str],
        output_name: str,
        filter: DataFilter,
        max_positions: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> str:
        """
        Build a processed dataset from PGN/ZST files.

        Args:
            source_paths: List of paths to source PGN/ZST files.
            output_name: Name for output dataset (without extension).
            filter: DataFilter for position filtering.
            max_positions: Maximum positions to extract.
            shuffle: Whether to shuffle positions.
            seed: Random seed for shuffling.

        Returns:
            Path to the created JSONL file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, f"{output_name}.jsonl")
        meta_path = os.path.join(self.output_dir, f"{output_name}_meta.json")

        extractor = DataExtractor(filter=filter)

        # Collect all positions
        positions = []
        for source_path in source_paths:
            if source_path.endswith(".zst"):
                iterator = extractor.extract_from_zst(
                    source_path,
                    max_positions=max_positions - len(positions) if max_positions else None,
                    progress=True,
                )
            else:
                iterator = extractor.extract_from_pgn(
                    source_path,
                    max_positions=max_positions - len(positions) if max_positions else None,
                    progress=True,
                )

            for pos in iterator:
                positions.append(pos)
                if max_positions and len(positions) >= max_positions:
                    break

            if max_positions and len(positions) >= max_positions:
                break

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(positions)

        # Write JSONL
        with open(output_path, "w") as f:
            for pos in positions:
                f.write(json.dumps(pos) + "\n")

        # Write metadata
        metadata = {
            "source": [os.path.basename(p) for p in source_paths],
            "num_positions": len(positions),
            "filters": {
                "eval_range_cp": list(filter.eval_range_cp),
                "min_ply": filter.min_ply,
                "min_game_length": filter.min_game_length,
                "require_eval": filter.require_eval,
            },
            "created": datetime.now().isoformat(),
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    def merge(
        self,
        input_paths: list[str],
        output_name: str,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> str:
        """
        Merge multiple JSONL datasets into one.

        Args:
            input_paths: List of paths to JSONL files.
            output_name: Name for output dataset.
            shuffle: Whether to shuffle merged positions.
            seed: Random seed for shuffling.

        Returns:
            Path to the merged JSONL file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, f"{output_name}.jsonl")

        # Collect all records
        records = []
        for input_path in input_paths:
            with open(input_path, "r") as f:
                for line in f:
                    records.append(json.loads(line.strip()))

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(records)

        # Write merged file
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        return output_path
