"""Regenerate tests/data/two_player_golden_fixtures.json.

The fixture is the byte-identity baseline for the two-player surface: the
golden test asserts that a fixed battery reproduces it exactly. Run this
script ONLY from a tree whose two-player behavior is known-good -- a fresh
checkout of the base commit -- never from a tree with unreviewed rating
changes. Regenerating from a changed tree silently re-baselines behavior
drift and defeats the fixture's purpose.

Usage (from the repository root, with the dev extras installed):

    python scripts/generate_two_player_goldens.py
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.test_two_player_goldens import FIXTURE_PATH, build_golden_states  # noqa: E402


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    payload = {"generated_from_commit": commit, "systems": build_golden_states()}
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {FIXTURE_PATH} from commit {commit}")


if __name__ == "__main__":
    main()
