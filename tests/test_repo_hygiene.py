"""Repository consistency: packaging metadata, CI matrix, README, static typing."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import autocall_pricer

ROOT = Path(__file__).resolve().parents[1]


def test_version_has_a_single_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = re.search(r"^\[project\]\n(.*?)(?=^\[)", pyproject, re.M | re.S).group(1)
    assert not re.search(r"^version\s*=", project, re.M), "version must not be duplicated in pyproject.toml"
    assert re.search(r'^dynamic\s*=\s*\[\s*"version"\s*\]', pyproject, re.M)
    assert 'version = { attr = "autocall_pricer.__version__" }' in pyproject
    assert re.fullmatch(r"\d+\.\d+\.\d+", autocall_pricer.__version__)


def test_ci_tests_python_3_14():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert '"3.14"' in ci


def test_readme_has_no_placeholder():
    assert "<user>" not in (ROOT / "README.md").read_text(encoding="utf-8")


def test_slow_marker_is_used():
    used = any("pytest.mark.slow" in p.read_text(encoding="utf-8")
               for p in (ROOT / "tests").glob("test_*.py") if p.name != Path(__file__).name)
    assert used


def test_tests_do_not_import_from_conftest():
    for p in (ROOT / "tests").glob("test_*.py"):
        assert not re.search(r"^from conftest import", p.read_text(encoding="utf-8"), re.M), p.name


@pytest.mark.skipif(shutil.which("mypy") is None and not
                    __import__("importlib.util").util.find_spec("mypy"), reason="mypy not installed")
def test_package_type_checks():
    out = subprocess.run([sys.executable, "-m", "mypy", str(ROOT / "src" / "autocall_pricer")],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr
