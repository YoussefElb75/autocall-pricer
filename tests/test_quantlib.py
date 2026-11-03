"""Monte Carlo vs QuantLib (see autocall_pricer.validation for the cases)."""

from dataclasses import replace

import pytest

pytest.importorskip("QuantLib")

from autocall_pricer import MonteCarloEngine  # noqa: E402
from autocall_pricer.validation import (  # noqa: E402
    _QLMarket,
    bonferroni_n_sigma,
    run_validation,
    validation_cases,
)

ENGINE = MonteCarloEngine(n_paths=200_000, seed=2026)
CASES = validation_cases()
N_SIGMA = bonferroni_n_sigma(len(CASES))


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_monte_carlo_matches_quantlib(case):
    row = replace(case.run(ENGINE, _QLMarket()), n_sigma=N_SIGMA)
    assert row.passed, (
        f"{row.case}: MC {row.mc_price:.6f} vs QL {row.ql_price:.6f} "
        f"(|diff| {row.abs_error:.6f} > tol {row.tolerance:.6f})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_validation_passes_for_other_seeds(seed):
    rows = run_validation(n_paths=100_000, seed=seed)
    assert all(r.passed for r in rows), [r.case for r in rows if not r.passed]


def test_validation_report_runs():
    rows = run_validation(n_paths=20_000)
    assert len(rows) == len(validation_cases())
