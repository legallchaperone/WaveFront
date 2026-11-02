"""``requirements/base.txt`` must pin exactly what ``pyproject.toml`` declares."""

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

_ROOT = Path(__file__).resolve().parents[1]


def _pins(lines: list[str]) -> dict[str, str]:
    pins = {}
    for line in lines:
        line = line.split(";")[0].split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name, version = line.split("==")
        pins[re.sub(r"[-_.]+", "-", name).lower()] = version.strip()
    return pins


def test_base_txt_matches_pyproject():
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    declared = _pins(project["dependencies"])
    frozen = _pins((_ROOT / "requirements/base.txt").read_text().splitlines())
    assert frozen == declared
