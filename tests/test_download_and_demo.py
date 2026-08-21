import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_and_demo import download_checkpoint


class DownloadAndDemoTests(unittest.TestCase):
    @patch("scripts.download_and_demo.hf_hub_download", return_value="cache/best.pt")
    def test_download_checkpoint_forwards_hub_coordinates(self, mocked_download):
        result = download_checkpoint("owner/model", "weights.pt", "v1")

        self.assertEqual(result, Path("cache/best.pt"))
        mocked_download.assert_called_once_with(
            repo_id="owner/model", filename="weights.pt", revision="v1"
        )


if __name__ == "__main__":
    unittest.main()
