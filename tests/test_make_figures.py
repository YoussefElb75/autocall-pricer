"""The README figure script runs end to end (images are not compared)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("QuantLib")  # the validation figure runs the QuantLib comparison

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _load_script():
    name = "make_figures"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up while the class is created
    spec.loader.exec_module(module)
    return module


def test_figure_script_runs_end_to_end(tmp_path):
    mf = _load_script()
    written = mf.main(tmp_path, mf.QUICK, verbose=False)
    assert [p.name for p in written] == [f"{name}.png" for name in mf.FIGURES]
    for path in written:
        assert path.parent == tmp_path
        data = path.read_bytes()
        assert data.startswith(PNG_SIGNATURE) and len(data) > 10_000, path.name


def test_committed_figures_match_the_script():
    mf = _load_script()
    committed = sorted(p.name for p in (ROOT / "docs" / "figures").glob("*.png"))
    assert committed == sorted(f"{name}.png" for name in mf.FIGURES)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in committed:
        assert f"docs/figures/{name}" in readme, f"{name} is not shown in the README"
