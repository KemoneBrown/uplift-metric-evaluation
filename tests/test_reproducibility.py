"""Unit tests for the reproducibility hashing helper."""
from src.reproducibility import file_hash


def test_file_hash_deterministic(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"identical content")
    assert file_hash(str(p)) == file_hash(str(p))


def test_file_hash_differs_on_change(tmp_path):
    p1 = tmp_path / "a.txt"; p1.write_bytes(b"one")
    p2 = tmp_path / "b.txt"; p2.write_bytes(b"two")
    assert file_hash(str(p1)) != file_hash(str(p2))