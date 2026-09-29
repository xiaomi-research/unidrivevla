"""Restore the original CoC JSONL files from the checked-in gzip archives."""

from pathlib import Path
import gzip
import hashlib
import shutil


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "alpamayo_coc_train_with_nav.jsonl": "5b21b67f321ac295efd3601d905b9514ea95afdf463811a19e2638bea1d76d3e",
    "alpamayo_coc_val_with_nav.jsonl": "7147c46286573c285dbecfd5deb0dd3832132baae1b707c4fe10e8eafb10f7ac",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for name, expected_hash in EXPECTED.items():
    archive = ROOT / f"{name}.gz"
    output = ROOT / name
    if output.exists() and sha256(output) == expected_hash:
        print(f"Already restored: {name}")
        continue

    temporary = ROOT / f"{name}.tmp"
    try:
        with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if sha256(temporary) != expected_hash:
            raise ValueError(f"Checksum mismatch: {archive.name}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Restored: {name}")
