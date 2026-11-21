"""Shared fixtures. The tests import the project modules from the repo root."""
import argparse
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if not (ROOT / "third_party" / "menagerie" / "universal_robots_ur5e").exists():
    pytest.exit("MuJoCo Menagerie missing: git clone https://github.com/google-deepmind/mujoco_menagerie "
                "third_party/menagerie", returncode=1)

from eval_place import add_scenario_args  # noqa: E402
from scene import build_model  # noqa: E402


def scenario_args(*argv):
    """eval_place.py's scenario flags, parsed exactly like the command line (new flags get their defaults)."""
    parser = argparse.ArgumentParser()
    add_scenario_args(parser)
    return parser.parse_args(list(argv))


@pytest.fixture
def model_data():
    """A fresh scene at the home pose."""
    return build_model()
