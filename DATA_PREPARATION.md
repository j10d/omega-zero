# Data Preparation Specification

## Overview

This component handles downloading, filtering, and processing chess game data for training. The goal is to create a local database of millions of positions with pre-computed evaluations.

## Data Source

**Primary source:** Lichess Monthly Game Database
- URL: https://database.lichess.org/
- Format: PGN files compressed with Zstandard (.pgn.zst)
- Size: Several GB per month (compressed), ~7x larger uncompressed
- Approximately 6% of games include Stockfish evaluations embedded as `[%eval]` annotations

**Alternative source:** Hugging Face pre-evaluated dataset
- URL: https://huggingface.co/datasets/Lichess/chess-position-evaluations
- 752M positions with FEN + centipawn + depth
- Already extracted, no PGN parsing needed
- But: no move played (policy target would need to be inferred or omitted)

**Recommended:** Lichess PGN files - provides both evaluation (value target) and move played (policy target).

## Filtering Strategy

Instead of filtering by player rating or time control, filter by **position quality** directly:

**Key insight:** Balanced positions (eval close to 0) indicate both sides played well to reach that position. Large evaluations indicate someone blundered - we don't want to train on those.

### Filtering Criteria

| Criterion | Value | Rationale |
|-----------|-------|-----------|
| Evaluation range | -200 to +200 cp | Balanced positions from well-played games |
| Minimum eval depth | 15+ | Consistent evaluation quality |
| Has evaluation | Required | Need value targets |
| Minimum game length | 20+ moves | Skip short games |
| Skip opening | First 8 moves | Avoid memorizing book moves |
| Variant | Standard only | Skip Chess960, etc. |

**Why this works:**
- A ±200 cp range means neither side has more than a 2-pawn advantage
- Positions with larger imbalances likely resulted from blunders
- Depth 15+ ensures evaluations are reasonably accurate
- Skipping early moves avoids overfitting to opening theory

## Evaluation Format in PGN

Lichess embeds evaluations as comments in the move text:

```pgn
1. e4 { [%eval 0.3] } 1... e5 { [%eval 0.28] } 2. Nf3 { [%eval 0.35] } 2... Nc6 { [%eval 0.32] }
```

**Evaluation format:**
- `[%eval 2.35]` = 235 centipawns advantage for White
- `[%eval -1.5]` = 150 centipawns advantage for Black
- `[%eval #4]` = White mates in 4
- `[%eval #-4]` = Black mates in 4
- Always from White's point of view

**Note:** Depth information may not always be available in PGN comments. The Hugging Face dataset includes depth.

## Storage Format

### Raw Downloaded Files

```
data/
├── raw/
│   ├── lichess_db_standard_rated_2024-01.pgn.zst
│   ├── lichess_db_standard_rated_2024-02.pgn.zst
│   └── ...
```

### Processed Dataset

Compact JSONL format (one position per line):

```
data/
├── processed/
│   ├── balanced_positions_5M.jsonl
│   └── balanced_positions_5M_meta.json
```

**Position record:**
```json
{"fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "move": "d2d3", "eval_cp": 35}
```

**Metadata file:**
```json
{
    "source": ["lichess_db_standard_rated_2024-01.pgn.zst"],
    "num_positions": 5000000,
    "filters": {
        "eval_range_cp": [-200, 200],
        "min_ply": 16,
        "min_game_length": 20
    },
    "created": "2024-01-15T10:30:00"
}
```

## Class Interface

### DataDownloader

Downloads Lichess database files.

```python
class DataDownloader:
    def __init__(
        self,
        output_dir: str = "data/raw"
    )
    
    def download_month(
        self,
        year: int,
        month: int,
        overwrite: bool = False
    ) -> str
    
    def list_available_months(self) -> list[tuple[int, int]]
    
    def get_download_url(self, year: int, month: int) -> str
```

### DataFilter

Filters positions based on evaluation criteria.

```python
class DataFilter:
    def __init__(
        self,
        eval_range_cp: tuple[int, int] = (-200, 200),
        min_ply: int = 16,
        min_game_length: int = 20,
        require_eval: bool = True
    )
    
    def filter_position(
        self,
        eval_cp: int | None,
        ply: int,
        game_length: int
    ) -> bool
        """Return True if position passes all filters."""
    
    def filter_game(self, game: chess.pgn.Game) -> bool
        """Return True if game is worth processing (has evals, long enough)."""
```

### DataExtractor

Extracts positions from filtered games.

```python
class DataExtractor:
    def __init__(
        self,
        filter: DataFilter
    )
    
    def extract_from_pgn(
        self,
        pgn_path: str,
        max_positions: int | None = None,
        progress: bool = True
    ) -> Iterator[dict]
        """
        Yield position records from a PGN file.
        
        Yields:
            {"fen": str, "move": str, "eval_cp": int}
        """
    
    def extract_from_zst(
        self,
        zst_path: str,
        max_positions: int | None = None,
        progress: bool = True
    ) -> Iterator[dict]
        """Extract directly from compressed file (streaming)."""
```

### DatasetBuilder

Builds the final processed dataset.

```python
class DatasetBuilder:
    def __init__(
        self,
        output_dir: str = "data/processed"
    )
    
    def build(
        self,
        source_paths: list[str],
        output_name: str,
        filter: DataFilter,
        max_positions: int | None = None,
        shuffle: bool = True,
        seed: int | None = None
    ) -> str
        """
        Build a processed dataset from PGN/ZST files.
        
        Returns:
            Path to the created JSONL file.
        """
    
    def merge(
        self,
        input_paths: list[str],
        output_name: str,
        shuffle: bool = True
    ) -> str
        """Merge multiple JSONL datasets into one."""
```

## Processing Pipeline

```
1. DOWNLOAD
   DataDownloader.download_month(2024, 1)
   → data/raw/lichess_db_standard_rated_2024-01.pgn.zst

2. FILTER + EXTRACT
   DataExtractor.extract_from_zst(
       "data/raw/lichess_db_standard_rated_2024-01.pgn.zst",
       max_positions=5000000
   )
   → yields {"fen": ..., "move": ..., "eval_cp": ...} records

3. BUILD DATASET
   DatasetBuilder.build(
       source_paths=["data/raw/lichess_db_standard_rated_2024-01.pgn.zst"],
       output_name="balanced_5M",
       filter=DataFilter(eval_range_cp=(-200, 200)),
       max_positions=5000000
   )
   → data/processed/balanced_5M.jsonl
```

## Streaming Processing

For large files (several GB), process without loading entire file into memory:

```python
import zstandard as zstd

def stream_pgn_from_zst(zst_path: str) -> Iterator[chess.pgn.Game]:
    """Stream games from compressed PGN without full decompression."""
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break
                yield game
```

## Evaluation Parsing

Extract centipawn evaluation from PGN comment:

```python
import re

def parse_eval(comment: str) -> int | None:
    """
    Parse [%eval X] from PGN comment.
    
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
        return None  # Skip mate positions
    
    # Convert to centipawns (eval is in pawns with decimal)
    return int(float(eval_str) * 100)
```

## Configuration Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| eval_range_cp | (-200, 200) | Balanced positions only |
| min_ply | 16 | Skip first 8 moves (16 half-moves) |
| min_game_length | 20 | Skip short games |
| require_eval | True | Only positions with evaluations |
| output_format | jsonl | JSON Lines format |

## Disk Space Requirements

**No full decompression needed.** The streaming approach reads directly from compressed `.pgn.zst` files, decompressing to memory (not disk). This keeps disk usage minimal:

| Item | Size |
|------|------|
| Compressed PGN file (.pgn.zst) | ~3-4 GB per month |
| Output JSONL (5M positions) | ~500 MB |
| Memory during processing | ~100 MB |
| Uncompressed PGN | **Not needed** |

You can delete the compressed source file after processing to reclaim space.

## Storage Estimates

For 5 million positions:
- JSONL format: ~500 MB
- Fields per position: ~100 bytes average

For reference:
- One month of Lichess data (compressed): ~3-4 GB
- ~6% of games have evaluations
- After filtering (balanced positions, skip openings): may need several months of data

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
data = [
    "zstandard",     # Zstd decompression
    "requests",      # HTTP downloads
    "tqdm",          # Progress bars
]
```

## Usage Example

```python
from data_preparation import DataDownloader, DataFilter, DatasetBuilder

# Download data
downloader = DataDownloader(output_dir="data/raw")
downloader.download_month(2024, 1)
downloader.download_month(2024, 2)

# Configure filter - balanced positions only
filter = DataFilter(
    eval_range_cp=(-200, 200),  # Within 2 pawns
    min_ply=16,                  # Skip opening
    min_game_length=20,
    require_eval=True
)

# Build dataset
builder = DatasetBuilder(output_dir="data/processed")
dataset_path = builder.build(
    source_paths=[
        "data/raw/lichess_db_standard_rated_2024-01.pgn.zst",
        "data/raw/lichess_db_standard_rated_2024-02.pgn.zst",
    ],
    output_name="balanced_5M",
    filter=filter,
    max_positions=5000000,
    shuffle=True,
    seed=42
)

print(f"Dataset created: {dataset_path}")
# → data/processed/balanced_5M.jsonl
```

## CLI Interface (Optional)

For convenience, a command-line interface:

```bash
# Download
python -m data_preparation download --year 2024 --month 1

# Build dataset
python -m data_preparation build \
    --source data/raw/lichess_db_standard_rated_2024-01.pgn.zst \
    --output balanced_5M \
    --eval-range -200 200 \
    --min-ply 16 \
    --max-positions 5000000

# Inspect dataset
python -m data_preparation info data/processed/balanced_5M.jsonl
```

## Implementation Notes

- Use streaming to handle large files without memory issues
- Progress bars for long operations (tqdm)
- Checkpoint/resume for interrupted downloads
- Validate FEN strings are legal positions
- Consider deduplicating identical FENs (same position from different games)
- Mate positions are excluded (outside balanced range)
- The move played is the policy target; the eval is the value target
