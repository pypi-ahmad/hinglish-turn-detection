import tempfile
import unittest
from pathlib import Path

import polars as pl

from scripts.transcribe_dataset import _resume_transcripts


class TranscriptCacheTests(unittest.TestCase):
    def test_resume_rejects_cache_from_different_asr_model(self):
        metadata = pl.DataFrame({"id": ["one"]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transcribed.parquet"
            metadata.with_columns(
                pl.lit("hello").alias("transcript"),
                pl.lit("old/model").alias("transcript_model"),
            ).write_parquet(output)

            with self.assertRaisesRegex(ValueError, "does not match requested model"):
                _resume_transcripts(metadata, output, "new/model")


if __name__ == "__main__":
    unittest.main()
