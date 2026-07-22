import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.download_checkpoint import download


class DownloadCheckpointTest(unittest.TestCase):
    def test_downloads_and_reuses_verified_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "weights" / "model.pth"
            source.write_bytes(b"verified checkpoint fixture")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            first = download(source.as_uri(), destination, digest)
            second = download(source.as_uri(), destination, digest)

            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_quarantines_invalid_existing_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "model.pth"
            source.write_bytes(b"correct")
            destination.write_bytes(b"incorrect")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            download(source.as_uri(), destination, digest)

            quarantined = list(root.glob("model.pth.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].read_bytes(), b"incorrect")
            self.assertEqual(destination.read_bytes(), b"correct")

    def test_rejects_download_with_wrong_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.bin"
            destination = root / "model.pth"
            source.write_bytes(b"wrong for requested digest")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                download(source.as_uri(), destination, "0" * 64)
            self.assertFalse(destination.exists())
            self.assertEqual(len(list(root.glob("model.pth.invalid-*"))), 1)


if __name__ == "__main__":
    unittest.main()
