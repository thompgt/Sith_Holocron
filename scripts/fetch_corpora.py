"""Fetch the third-party screenplay corpora this project indexes.

The dialogue half of the hybrid retriever comes from two public repositories that
are *not* vendored here. They used to be committed as bare gitlinks with no
`.gitmodules`, which meant a fresh clone got two empty directories, the index
built lore-only, and the "hybrid" half of the product vanished with no error at
all. This script is the replacement: run it once after cloning.

    python scripts/fetch_corpora.py

It is idempotent — an already-populated corpus is left alone unless you pass
`--force`. Requires `git` on PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"


class Corpus:
    def __init__(
        self,
        name: str,
        url: str,
        sentinel: str,
        required: bool,
        note: str,
        sparse: list[str] | None = None,
    ):
        self.name = name
        self.url = url
        # A path that must exist for the corpus to count as present. Checking a
        # real file rather than the directory matters because the old gitlinks
        # left the directories themselves behind.
        self.sentinel = sentinel
        self.required = required
        self.note = note
        # Subdirectories to check out, for repos that cannot be checked out whole
        # on every platform. None means a full checkout.
        self.sparse = sparse

    @property
    def path(self) -> Path:
        return RAW_DIR / self.name

    def is_present(self) -> bool:
        return (self.path / self.sentinel).exists()


CORPORA = [
    Corpus(
        name="star-wars-scripts",
        url="https://github.com/gastonstat/StarWars.git",
        sentinel="Text_files/EpisodeIV_dialogues.txt",
        required=True,
        note="original-trilogy dialogue; src/main.py reads Episode IV from it",
    ),
    Corpus(
        name="prequel-scripts",
        url="https://github.com/jcwieme/data-scripts-star-wars.git",
        sentinel="5. Data CSV/star_wars_1_data.csv",
        required=False,
        # The prequel CSVs the app actually indexes are vendored under
        # data/raw/prequel-csv/, so nothing breaks without this. It is the
        # provenance of those CSVs, worth fetching to re-derive or extend them.
        note="upstream provenance of data/raw/prequel-csv/; not read at runtime",
        # A full checkout fails on Windows: the repo carries a macOS resource
        # file literally named "Icon?", and "?" is not a legal Windows filename,
        # so git aborts the whole working tree with "invalid path". Checking out
        # only the CSV directory sidesteps it and is all this corpus is for.
        sparse=["5. Data CSV"],
    ),
]


def _on_rm_error(func, path, exc_info):
    """Clear the read-only bit and retry.

    Git marks objects and pack files read-only, and on Windows os.unlink refuses
    to delete a read-only file, so a plain rmtree over a cloned corpus fails with
    WinError 5.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise


def rmtree(path: Path) -> None:
    shutil.rmtree(path, onexc=_on_rm_error)


def clone(corpus: Corpus, force: bool) -> bool:
    """Clone one corpus. Returns True if the corpus is usable afterwards."""
    if corpus.is_present() and not force:
        print(f"  {corpus.name}: already present, skipping")
        return True

    if corpus.path.exists():
        print(f"  {corpus.name}: removing existing directory")
        rmtree(corpus.path)

    corpus.path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {corpus.name}: cloning {corpus.url}")

    clone_cmd = ["git", "clone", "--depth", "1"]
    if corpus.sparse:
        # Clone the history but write no working tree, then check out only the
        # wanted paths. sparse-checkout does not help here: it does not constrain
        # the *initial* checkout, so git still tries to write the illegal path.
        clone_cmd.append("--no-checkout")
    clone_cmd += [corpus.url, str(corpus.path)]

    steps = [clone_cmd]
    if corpus.sparse:
        steps.append(
            ["git", "-C", str(corpus.path), "checkout", "HEAD", "--", *corpus.sparse]
        )

    for cmd in steps:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {corpus.name}: FAILED\n{result.stderr.strip()}", file=sys.stderr)
            return False

    # Drop the nested .git so the corpus does not read as a gitlink to the
    # parent repo -- that is the exact failure mode this script exists to undo.
    nested_git = corpus.path / ".git"
    if nested_git.is_dir():
        rmtree(nested_git)
    elif nested_git.exists():
        nested_git.unlink()

    if not corpus.is_present():
        print(
            f"  {corpus.name}: cloned but {corpus.sentinel} is missing -- "
            "upstream layout may have changed",
            file=sys.stderr,
        )
        return False

    print(f"  {corpus.name}: ok")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-clone even if the corpus is already present",
    )
    args = parser.parse_args()

    if shutil.which("git") is None:
        print("git not found on PATH", file=sys.stderr)
        return 1

    print(f"Fetching {len(CORPORA)} corpora into {RAW_DIR}")
    failed = []
    for corpus in CORPORA:
        print(f"\n{corpus.name} -- {corpus.note}")
        if not clone(corpus, args.force):
            failed.append(corpus)

    required_failed = [c for c in failed if c.required]
    optional_failed = [c for c in failed if not c.required]

    if optional_failed:
        print(
            "\nOptional corpora unavailable (not read at runtime): "
            + ", ".join(c.name for c in optional_failed)
        )

    if required_failed:
        print(
            "\nRequired corpora failed to fetch: "
            + ", ".join(c.name for c in required_failed)
            + "\nThe index would build lore-only and persona voice matching "
            "would find nothing.",
            file=sys.stderr,
        )
        return 1

    print("\nRequired corpora present. Next: python -m src.main (builds the index).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
