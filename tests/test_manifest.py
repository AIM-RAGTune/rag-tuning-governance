from pathlib import Path

from square_sim.utils.hashing import sha256_file, write_checksums


def test_manifest_checksum_creation(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("square\n", encoding="utf-8")
    checksum = sha256_file(p)
    assert len(checksum) == 64
    checksums = write_checksums([p], tmp_path / "checksums.sha256")
    assert str(p) in checksums
    assert (tmp_path / "checksums.sha256").exists()

