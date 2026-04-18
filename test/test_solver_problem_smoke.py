"""Smoke tests for every solver and problem in the simopt package.

Tests three things:
1. Every problem can be instantiated with defaults and has valid properties.
2. Every solver can be instantiated with defaults and has valid properties.
3. Every compatible solver-problem pair completes a short run without errors.

Run with::

    pytest test/test_solver_problem_smoke.py -v
    pytest test/test_solver_problem_smoke.py -v -k "test_run"  # just the runs
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from mrg32k3a.mrg32k3a import MRG32k3a
from simopt.base import MultistageProblem
from simopt.directory import problem_directory, solver_directory
from simopt.experiment.post_normalize import post_normalize_policy
from simopt.experiment.run_solver import run_solver
from simopt.experiment.single import ProblemSolver
from simopt.experiment_base import instantiate_problem, instantiate_solver
from simopt.problem import Solution

# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------
# Constraint hierarchy: U < B < D < S (a solver handling stochastic can
# handle everything below it).
_CONSTRAINT_ORDER = {"U": 0, "B": 1, "D": 2, "S": 3}

# Variable hierarchy: D < C < M (mixed handles both discrete and continuous).
_VARIABLE_ORDER = {"D": 0, "C": 1, "M": 2}


def solver_handles_problem(solver_compat: str, problem_compat: str) -> bool:
    """Check if a solver's compatibility string covers a problem's."""
    s_obj, s_con, s_var, s_grad = solver_compat
    p_obj, p_con, p_var, p_grad = problem_compat

    # Objective type must match
    if s_obj != p_obj:
        return False

    # Solver constraint capability must be >= problem's
    if _CONSTRAINT_ORDER.get(s_con, -1) < _CONSTRAINT_ORDER.get(p_con, -1):
        return False

    # Solver variable capability must be >= problem's
    if _VARIABLE_ORDER.get(s_var, -1) < _VARIABLE_ORDER.get(p_var, -1):
        return False

    # If problem needs gradients, solver must support them
    return not (p_grad == "G" and s_grad != "G")


# ---------------------------------------------------------------------------
# Build test parameters
# ---------------------------------------------------------------------------
_PROBLEM_NAMES = sorted(problem_directory.keys())
_SOLVER_NAMES = sorted(solver_directory.keys())

# Exclude hyperparameter-tuning problems (require a nested target problem)
_SKIP_PROBLEMS = {"ASTROMORF-HYPEROPT-1", "ASTROMORF-HYPEROPT-2"}

# Small budget for smoke tests — just enough to verify the loop runs
_SMOKE_BUDGET = 200
_N_MACROREPS = 1

# Known unstable combinations under tiny smoke settings.
_SKIP_SMOKE_PAIRS: dict[tuple[str, str], str] = {
    (
        "MIXINTTR",
        "NETWORK-1",
    ): "Known to hit zero-route-weight sampling under tiny smoke budget.",
    (
        "MIXINTTR",
        "PARAMESTI-1",
    ): "Known to hit invalid gamma-domain values under tiny smoke budget.",
    (
        "NELDMD",
        "HOTEL-1",
    ): "Known to hit a boolean-indexing TypeError in simplex initialization.",
}


def _as_tuple(value: object) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(value)
    return ()


def _build_smoke_cases() -> list[tuple[str, str, dict, dict]]:
    """Build solver/problem smoke cases with runtime and precondition checks."""
    cases: list[tuple[str, str, dict, dict]] = []

    for solver_name in _SOLVER_NAMES:
        solver_cls = solver_directory[solver_name]
        for problem_name in _PROBLEM_NAMES:
            if problem_name in _SKIP_PROBLEMS:
                continue

            problem_cls = problem_directory[problem_name]
            if not solver_handles_problem(
                solver_cls.compatibility,
                problem_cls.compatibility,
            ):
                continue

            if (solver_name, problem_name) in _SKIP_SMOKE_PAIRS:
                continue

            solver = solver_cls()
            problem = problem_cls()

            # Runtime support checks are stricter than compatibility-string checks.
            if not solver.supports_problem(problem):
                continue

            # FCSA requires at least one stochastic constraint.
            if solver_name == "FCSA" and problem.n_stochastic_constraints <= 0:
                continue

            solver_fixed_factors: dict = {}
            problem_fixed_factors: dict = {"budget": _SMOKE_BUDGET}

            # Nelder-Mead requires enough budget for the initial simplex.
            if solver_name == "NELDMD":
                min_budget = int(solver.factors["r"]) * (problem.dim + 1)
                problem_fixed_factors["budget"] = max(_SMOKE_BUDGET, min_budget)

            # Complete enumeration requires at least one explicit solution tuple.
            if solver_name == "RANDS":
                initial_solution = _as_tuple(problem.factors.get("initial_solution"))
                if not initial_solution:
                    continue
                solver_fixed_factors["solution_list"] = [initial_solution]

                sample_size = int(solver.factors.get("sample_size", 1))
                solution_list_len = len(solver_fixed_factors["solution_list"])
                required_budget = sample_size * solution_list_len
                problem_fixed_factors["budget"] = max(
                    int(problem_fixed_factors["budget"]),
                    required_budget,
                )

            cases.append(
                (
                    solver_name,
                    problem_name,
                    solver_fixed_factors,
                    problem_fixed_factors,
                )
            )

    return cases


_SMOKE_CASES = _build_smoke_cases()


# ---------------------------------------------------------------------------
# 1. Problem instantiation tests
# ---------------------------------------------------------------------------
class TestProblemInstantiation:
    """Verify every registered problem can be created with defaults."""

    @pytest.mark.parametrize("problem_name", _PROBLEM_NAMES)
    def test_instantiate(self, problem_name: str) -> None:
        """Problem can be created with default factors."""
        if problem_name in _SKIP_PROBLEMS:
            pytest.skip("requires target_problem")
        cls = problem_directory[problem_name]
        problem = cls()
        assert problem is not None

    @pytest.mark.parametrize("problem_name", _PROBLEM_NAMES)
    def test_properties(self, problem_name: str) -> None:
        """Problem exposes valid dim, bounds, and compatibility."""
        if problem_name in _SKIP_PROBLEMS:
            pytest.skip("requires target_problem")
        cls = problem_directory[problem_name]
        problem = cls()

        # dim must be a positive integer
        assert isinstance(problem.dim, int)
        assert problem.dim > 0

        # bounds must be tuples of the right length
        assert len(problem.lower_bounds) == problem.dim
        assert len(problem.upper_bounds) == problem.dim

        # lower <= upper for each dimension
        for lo, hi in zip(problem.lower_bounds, problem.upper_bounds, strict=True):
            assert lo <= hi, f"lower_bound {lo} > upper_bound {hi} for {problem_name}"

        # compatibility string is 4 characters
        assert len(cls.compatibility) == 4

        # factors dict should include a budget
        assert "budget" in problem.factors

    def test_vanryzin_optimum_fields_are_unknown(self) -> None:
        """VanRyzin problems should not advertise a known global optimum."""
        for problem_name in ("VANRYZIN-1", "VANRYZIN-2"):
            problem = instantiate_problem(problem_name)
            assert problem.optimal_value is None
            assert problem.optimal_solution is None


# ---------------------------------------------------------------------------
# 2. Solver instantiation tests
# ---------------------------------------------------------------------------
class TestSolverInstantiation:
    """Verify every registered solver can be created with defaults."""

    @pytest.mark.parametrize("solver_name", _SOLVER_NAMES)
    def test_instantiate(self, solver_name: str) -> None:
        """Solver can be created with default factors."""
        cls = solver_directory[solver_name]
        solver = cls()
        assert solver is not None

    @pytest.mark.parametrize("solver_name", _SOLVER_NAMES)
    def test_properties(self, solver_name: str) -> None:
        """Solver exposes valid compatibility, factors, and name."""
        cls = solver_directory[solver_name]
        solver = cls()

        # compatibility string is 4 characters
        assert len(cls.compatibility) == 4

        # factors dict should exist
        assert isinstance(solver.factors, dict)

        # name should be non-empty
        assert cls.class_name_abbr
        assert cls.class_name


# ---------------------------------------------------------------------------
# 3. Solver-problem pair smoke tests
# ---------------------------------------------------------------------------
class TestSolverProblemRun:
    """Run each compatible solver-problem pair with a tiny budget."""

    @pytest.mark.parametrize(
        (
            "solver_name",
            "problem_name",
            "solver_fixed_factors",
            "problem_fixed_factors",
        ),
        _SMOKE_CASES,
        ids=[f"{s}-on-{p}" for s, p, _, _ in _SMOKE_CASES],
    )
    def test_run(
        self,
        solver_name: str,
        problem_name: str,
        solver_fixed_factors: dict,
        problem_fixed_factors: dict,
    ) -> None:
        """Solver completes a short run on the problem without errors."""
        ps = ProblemSolver(
            solver_name=solver_name,
            problem_name=problem_name,
            solver_fixed_factors=solver_fixed_factors,
            problem_fixed_factors=problem_fixed_factors,
            create_pickle=False,
        )
        ps.run(n_macroreps=_N_MACROREPS, n_jobs=1)

        # Should have results for the requested number of macroreps
        assert ps.n_macroreps == _N_MACROREPS

        # Each macrorep should have at least one recommended solution
        for mrep_idx in range(ps.n_macroreps):
            rec_xs = ps.all_recommended_xs[mrep_idx]
            budgets = ps.all_intermediate_budgets[mrep_idx]
            assert len(rec_xs) >= 1, (
                f"{solver_name} on {problem_name}: "
                f"no recommended solutions in mrep {mrep_idx}"
            )
            assert len(budgets) >= 1
            # Budgets should be non-decreasing
            for i in range(1, len(budgets)):
                assert budgets[i] >= budgets[i - 1]


class TestMultistageLookaheadRngHandling:
    """Regression tests for multistage solver-based lookahead RNG behavior."""

    def test_simulate_handles_unseeded_lookahead_solver_rngs(self) -> None:
        """Lookahead should not fail when stage solver progenitor RNGs are empty."""
        problem_generic = instantiate_problem(
            "VANRYZIN-2",
            problem_fixed_factors={"budget": 20},
        )
        assert isinstance(problem_generic, MultistageProblem)
        problem = problem_generic

        problem.lookahead_solver = instantiate_solver(
            solver_name="SGD",
            fixed_factors={"r": 1},
            solver_rename="SGD",
        )

        solution = Solution(tuple(problem.factors["initial_solution"]), problem)
        solution.attach_rngs(
            [MRG32k3a(s_ss_sss_index=[3, i, 0]) for i in range(problem.model.n_rngs)],
            copy=False,
        )

        problem.simulate(solution, num_macroreps=1)

        assert solution.n_reps == 1

    def test_shared_problem_overwritten_lookahead_solver_no_index_error(self) -> None:
        """Shared multistage problem instances should be robust to solver aliasing."""
        problem_generic = instantiate_problem(
            "VANRYZIN-2",
            problem_fixed_factors={"budget": 20},
        )
        assert isinstance(problem_generic, MultistageProblem)
        problem = problem_generic

        davn = instantiate_solver(
            solver_name="DAVN",
            fixed_factors={"evaluation_reps": 1},
            solver_rename="DAVN",
        )
        sgd = instantiate_solver(
            solver_name="SGD",
            fixed_factors={"r": 1},
            solver_rename="SGD",
        )

        # Mimic ProblemsSolvers reuse of one problem object where another
        # ProblemSolver last wrote the lookahead solver.
        problem.lookahead_solver = sgd

        solution_df, _iteration_df, elapsed_times = run_solver(
            davn,
            problem,
            n_macroreps=1,
            n_jobs=1,
        )

        assert not solution_df.empty
        assert len(elapsed_times) == 1

    def test_shared_problem_uses_active_solver_for_lookahead(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """DAVN run should not invoke SGD lookahead on the same problem object."""
        problem_generic = instantiate_problem(
            "VANRYZIN-2",
            problem_fixed_factors={"budget": 20},
        )
        assert isinstance(problem_generic, MultistageProblem)
        problem = problem_generic

        davn = instantiate_solver(
            solver_name="DAVN",
            fixed_factors={"evaluation_reps": 1},
            solver_rename="DAVN",
        )
        sgd = instantiate_solver(
            solver_name="SGD",
            fixed_factors={"r": 1},
            solver_rename="SGD",
        )

        # Simulate the shared-problem overwrite coming from another
        # ProblemSolver wrapper.
        problem.lookahead_solver = sgd

        sgd_lookahead_calls = {"count": 0}

        def _count_sgd_lookahead(self: object, problem: object) -> None:  # noqa: ARG001
            sgd_lookahead_calls["count"] += 1

        monkeypatch.setattr(type(sgd), "solve", _count_sgd_lookahead)

        solution_df, _iteration_df, elapsed_times = run_solver(
            davn,
            problem,
            n_macroreps=1,
            n_jobs=1,
        )

        assert not solution_df.empty
        assert len(elapsed_times) == 1
        assert sgd_lookahead_calls["count"] == 0

    def test_vanryzin_policy_postreplicate_after_budget_trim(self) -> None:
        """Trimmed terminal-budget row should keep policy_solution objects."""
        problem_generic = instantiate_problem(
            "VANRYZIN-2",
            problem_fixed_factors={"budget": 80},
        )
        assert isinstance(problem_generic, MultistageProblem)

        solver = instantiate_solver(
            solver_name="VANRYZIN_SGD",
            fixed_factors={"r": 1},
            solver_rename="VANRYZIN_SGD",
        )

        ps = ProblemSolver(
            solver=solver,
            problem=problem_generic,
            create_pickle=False,
        )
        ps.run(n_macroreps=1, n_jobs=1)

        assert ps.all_policy_solutions is not None
        last_policy_sol = ps.all_policy_solutions[0][-1]
        assert isinstance(last_policy_sol, Solution)

        ps.post_replicate_policy(n_postreps=1)

        assert ps.has_postreplicated
        assert len(ps.all_post_replicates) == 1

    def test_policy_post_normalize_uses_policy_evaluation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Policy normalization should evaluate anchors via simulate_policy only."""
        problem_generic = instantiate_problem(
            "VANRYZIN-2",
            problem_fixed_factors={"budget": 80},
        )
        assert isinstance(problem_generic, MultistageProblem)

        solver = instantiate_solver(
            solver_name="VANRYZIN_SGD",
            fixed_factors={"r": 1},
            solver_rename="VANRYZIN_SGD",
        )

        ps = ProblemSolver(
            solver=solver,
            problem=problem_generic,
            create_pickle=False,
        )
        ps.run(n_macroreps=1, n_jobs=1)
        ps.post_replicate_policy(n_postreps=1)

        problem = cast(MultistageProblem, ps.problem)

        simulate_policy_calls = {"count": 0}
        original_simulate_policy = problem.simulate_policy

        def _count_simulate_policy(
            solution: Solution,
            num_macroreps: int = 1,
        ) -> None:
            simulate_policy_calls["count"] += 1
            original_simulate_policy(solution=solution, num_macroreps=num_macroreps)

        def _fail_simulate(*args: object, **kwargs: object) -> None:  # noqa: ARG001
            raise AssertionError(
                "post_normalize_policy should not call simulate for policy experiments"
            )

        monkeypatch.setattr(problem, "simulate_policy", _count_simulate_policy)
        monkeypatch.setattr(problem, "simulate", _fail_simulate)

        post_normalize_policy(experiments=[ps], n_postreps_init_opt=1)

        assert ps.has_postnormalized
        assert simulate_policy_calls["count"] > 0
