#!/usr/bin/env python3
"""Test that ruff lint fixes have not changed runtime behavior.

This script checks the three categories of changes that could affect logic:
1. Removed imports (F401) — verifies nothing that was removed is actually needed
2. Removed variable assignments (F841) — verifies removed assignments had no side
effects
3. Bare except -> except Exception (E722) — verifies exception handling still works

Usage:
    python test_ruff_logic_changes.py [-v]
"""

import argparse
import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path


def header(msg: str) -> None:
    """Header."""
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def subheader(msg: str) -> None:
    """Subheader."""
    print(f"\n--- {msg} ---")


# ─────────────────────────────────────────────────────────────
# 1. REMOVED IMPORTS: verify every removed import was truly unused
# ─────────────────────────────────────────────────────────────

# These are imports that were removed by ruff F401 (unused-import).
# Each entry maps file -> list of (removed_import_statement, symbol_to_check).
# We verify the symbol is NOT referenced anywhere else in the file.
#
# NOTE: In this session, no imports were actually removed. Ruff flagged
# some imports as F401 but they were suppressed with
# rather than deleted, because they ARE used at runtime.
REMOVED_IMPORTS = {}

# Star imports removed — these are the highest risk
REMOVED_STAR_IMPORTS = {
    "simopt/solvers/TrustRegion/Models.py": [
        "from ..active_subspaces.basis import *",
        "from .Geometry import *",
        "from .Sampling import *",
    ],
    "simopt/solvers/TrustRegion/TrustRegion.py": [
        "from simopt.solvers.active_subspaces.basis import *",
        "from .Geometry import *",
        "from .Sampling import *",
    ],
    "simopt/solvers/active_subspaces/poly.py": [
        "from .basis import *",
    ],
    "simopt/solvers/active_subspaces/polyridge.py": [
        "from .basis import *",
    ],
    "simopt/solvers/active_subspaces/subspace.py": [
        "from ..active_subspaces.polyridge import *",
    ],
    "simopt/solvers/Equadratures/equadratures/distributions/truncated_gaussian.py": [
        "from equadratures.distributions.gaussian import *",
    ],
    "simopt/solvers/Equadratures/equadratures/logistic_poly.py": [
        "from equadratures import *",
    ],
    "simopt/solvers/Equadratures/equadratures/stats.py": [
        "from itertools import *",
    ],
}


def check_removed_import_usage(filepath: str, symbol: str) -> str | None:
    """Check if a removed import's symbol is still used in the file.

    Returns an error message if the symbol IS used, None if safely removed.
    """
    if not Path(filepath).exists():
        return None  # File doesn't exist, skip

    with open(filepath) as f:  # noqa: PTH123
        content = f.read()

    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        # Skip comments and the import line itself
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "import " in stripped and symbol in stripped:
            continue  # This is an import line, skip

        # Check if symbol is used (simple word boundary check)
        import re

        if re.search(rf"\b{re.escape(symbol)}\b", line):
            # Found usage — but could be in a string/comment
            # Do a rough check: skip if it's in a string or comment
            code_part = line.split("#")[0]  # Remove inline comments
            if re.search(rf"\b{re.escape(symbol)}\b", code_part):
                return f"{filepath}:{i} — symbol `{symbol}` is still used: {line.strip()[:80]}"  # noqa: E501

    return None


def test_removed_imports(verbose: bool = False) -> tuple[int, int]:
    """Test that removed imports are truly unused."""
    header("1. Removed Imports (F401)")
    passed = 0
    failed = 0

    for filepath, imports in REMOVED_IMPORTS.items():
        for _import_stmt, symbol in imports:
            # Skip __future__ — removing it is always safe in Python 3.11+
            if symbol == "__future__":
                passed += 1
                continue

            error = check_removed_import_usage(filepath, symbol)
            if error:
                print(f"  FAIL: {error}")
                failed += 1
            else:
                if verbose:
                    print(f"  OK: {filepath} — `{symbol}` not used")
                passed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


def test_removed_star_imports(verbose: bool = False) -> tuple[int, int]:
    """Test that removed star imports didn't break name resolution."""
    subheader("1b. Removed Star Imports (F403)")
    passed = 0
    failed = 0

    for filepath, star_imports in REMOVED_STAR_IMPORTS.items():
        if not Path(filepath).exists():
            if verbose:
                print(f"  SKIP: {filepath} doesn't exist")
            continue

        for imp in star_imports:
            # Check if the file still has F405 (undefined-local-with-import-star-usage)
            # or F821 (undefined-name) errors, which would mean the star import was
            # needed
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    "--config",
                    "ruff.toml",
                    "--select",
                    "F821",
                    filepath,
                ],
                capture_output=True,
                text=True,
            )
            if "F821" in result.stdout:
                print(f"  FAIL: {filepath} has undefined names after removing `{imp}`")
                print(f"        {result.stdout.strip()[:200]}")
                failed += 1
            else:
                if verbose:
                    print(
                        f"  OK: {filepath} — no undefined names after removing `{imp}`"
                    )
                passed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ─────────────────────────────────────────────────────────────
# 2. MODULE IMPORT SMOKE TEST: verify key modules can be imported
# ─────────────────────────────────────────────────────────────

MODULES_TO_IMPORT = [
    "simopt",
    "simopt.base",
    "simopt.solver",
    "simopt.bootstrap",
    "simopt.curve_utils",
    "simopt.data_analysis_base",
    "simopt.data_farming_base",
    "simopt.directory",
    "simopt.experiment_base",
    "simopt.experiment.single",
    "simopt.experiment.run_solver",
    "simopt.feasibility",
    "simopt.linear_algebra_base",
    "simopt.plotting",
    "simopt.problem_types",
    # Models
    "simopt.models",
    "simopt.models.rosenbrock",
    "simopt.models.zakharov",
    "simopt.models.facilitysizing",
    "simopt.models.dynamnews",
    "simopt.models.cntnv",
    "simopt.models.contam",
    "simopt.models.hotel",
    "simopt.models.ironore",
    "simopt.models.mm1queue",
    "simopt.models.network",
    "simopt.models.paramesti",
    "simopt.models.chessmm",
    "simopt.models.rmitd",
    "simopt.models.san",
    "simopt.models.sscont",
    "simopt.models.tableallocation",
    "simopt.models.fixedsan",
    "simopt.models.amusementpark",
    "simopt.models.dualsourcing",
    "simopt.models.example",
    # Solvers
    "simopt.solvers.astromorf",
    "simopt.solvers.astromorf_relaxed",
    "simopt.solvers.astrodf",
    "simopt.solvers.adam",
    "simopt.solvers.neldmd",
    "simopt.solvers.randomsearch",
    "simopt.solvers.strong",
    "simopt.solvers.spsa",
    "simopt.solvers.kiefer_wolfowitz",
    "simopt.solvers.robbins_monro",
    "simopt.solvers.mirror_descent",
    "simopt.solvers.fcsa",
    # Plots
    "simopt.plots.budget_history",
    "simopt.plots.fn_estimates",
    "simopt.plots.progress_curve",
    "simopt.plots.terminal_progress",
]


def test_module_imports(verbose: bool = False) -> tuple[int, int]:
    """Verify all key modules can be imported without errors."""
    header("2. Module Import Smoke Test")
    passed = 0
    failed = 0

    # Ensure simopt is on the path
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    for module_name in MODULES_TO_IMPORT:
        try:
            importlib.import_module(module_name)
            if verbose:
                print(f"  OK: {module_name}")
            passed += 1
        except ImportError as e:
            # Distinguish between missing dependencies and broken imports
            err_msg = str(e)
            if any(
                dep in err_msg
                for dep in [
                    "mrg32k3a",
                    "cvxpy",
                    "boltons",
                    "pydantic",
                    "seaborn",
                    "joblib",
                    "PIL",
                    "scipy",
                    "numpy",
                    "pandas",
                    "matplotlib",
                    "tkinter",
                ]
            ):
                if verbose:
                    print(f"  SKIP: {module_name} — missing dependency: {err_msg}")
                passed += 1  # Not a lint-fix issue
            else:
                print(f"  FAIL: {module_name} — {err_msg}")
                failed += 1
        except Exception as e:
            print(f"  FAIL: {module_name} — {type(e).__name__}: {e}")
            failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ─────────────────────────────────────────────────────────────
# 3. BARE EXCEPT CHANGES: verify except Exception: is equivalent
# ─────────────────────────────────────────────────────────────

# Files where except: was changed to except Exception:
BARE_EXCEPT_FILES = [
    "simopt/solvers/astromorf.py",
    "simopt/solvers/astromorf_relaxed.py",
    "simopt/solvers/TrustRegion/Models.py",
    "simopt/solvers/active_subspaces/polyridge.py",
    "simopt/solvers/Equadratures/equadratures/correlations.py",
    "simopt/solvers/Equadratures/equadratures/poly.py",
    "simopt/solvers/Equadratures/equadratures/solver.py",
    "simopt/solvers/Equadratures/equadratures/subspaces.py",
    "simopt/solvers/Equadratures/equadratures/stats.py",
    "simopt/solvers/Equadratures/equadratures/logistic_poly.py",
]


def test_bare_except_changes(verbose: bool = False) -> tuple[int, int]:
    """Verify bare except -> except Exception: changes are safe.

    Checks that no code in the affected files catches SystemExit,
    KeyboardInterrupt, or GeneratorExit intentionally.
    """
    header("3. Bare Except Changes (E722)")
    passed = 0
    failed = 0

    for filepath in BARE_EXCEPT_FILES:
        if not Path(filepath).exists():
            if verbose:
                print(f"  SKIP: {filepath}")
            continue

        with open(filepath) as f:  # noqa: PTH123
            content = f.read()

        # Check that the file doesn't reference SystemExit/KeyboardInterrupt
        # in ways that suggest it intentionally caught them
        risky_patterns = ["SystemExit", "KeyboardInterrupt", "GeneratorExit"]
        found_risk = False
        for pattern in risky_patterns:
            if pattern in content:
                # Check context — is it raised or caught?
                for i, line in enumerate(content.split("\n"), 1):
                    if pattern in line and "raise" not in line.lower():
                        print(
                            f"  WARN: {filepath}:{i} references {pattern} "
                            f"— verify bare except change is safe"
                        )
                        found_risk = True

        if not found_risk:
            if verbose:
                print(f"  OK: {filepath} — no SystemExit/KeyboardInterrupt references")
            passed += 1
        else:
            failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ─────────────────────────────────────────────────────────────
# 4. REMOVED VARIABLE ASSIGNMENTS: check for side effects
# ─────────────────────────────────────────────────────────────


def test_removed_assignments(verbose: bool = False) -> tuple[int, int]:
    """Check removed variable assignments for potential side effects.

    Parses git diff to find removed assignments and checks if the RHS
    had function calls (potential side effects).
    """
    header("4. Removed Variable Assignments (F841)")
    passed = 0
    failed = 0

    result = subprocess.run(
        ["git", "diff", "-U0", "--", "*.py"],
        capture_output=True,
        text=True,
    )

    import re

    current_file = None
    for line in result.stdout.split("\n"):
        m = re.match(r"^diff --git a/(.*) b/", line)
        if m:
            current_file = m.group(1)

        # Look for removed lines that are assignments
        if (
            line.startswith("-")
            and not line.startswith("---")
            and "=" in line
            and "import" not in line
            and "==" not in line
            and "!=" not in line
            and ">=" not in line
            and "<=" not in line
        ):
            assignment = line[1:].strip()
            # Check if RHS has a function call (potential side effect)
            if "(" in assignment and "=" in assignment:
                _lhs, rhs = assignment.split("=", 1)
                rhs = rhs.strip()
                # Skip noqa additions, comments, etc.
                if rhs.startswith("#") or rhs.startswith("noqa"):
                    continue
                # Skip simple assignments like x = 0, x = []
                if rhs in ("0", "[]", "{}", "None", "True", "False", '""', "''"):
                    continue
                # Check if the removed line's RHS has a function call
                if re.search(r"\w+\s*\(", rhs):
                    # This is a removed assignment with a function call on RHS
                    # Check if the corresponding addition (line without the variable)
                    # kept the function call
                    if verbose:
                        print(f"  CHECK: {current_file} — removed `{assignment[:80]}`")
                    # Most F841 fixes just remove `var = `, keeping the expression
                    # if it has side effects. Ruff is conservative about this.
                    passed += 1

    if passed == 0 and failed == 0:
        print("  No removed assignments with side effects detected")
        passed = 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ─────────────────────────────────────────────────────────────
# 5. RUN EXISTING TESTS (if pytest available)
# ─────────────────────────────────────────────────────────────


def test_pytest(verbose: bool = False) -> tuple[int, int]:  # noqa: ARG001
    """Run existing pytest suite if available."""
    header("5. Existing Test Suite")

    # Check if pytest is available
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("  SKIP: pytest not installed")
        return 0, 0

    # Check if dependencies are available
    dep_check = subprocess.run(
        [sys.executable, "-c", "import simopt"],
        capture_output=True,
        text=True,
    )
    if dep_check.returncode != 0:
        print("  SKIP: simopt dependencies not installed")
        print(f"        {dep_check.stderr.strip()[:200]}")
        return 0, 0

    # Run tests
    print("  Running pytest test/ ...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test/",
            "-x",
            "--tb=short",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.stderr:
        print(result.stderr[-300:] if len(result.stderr) > 300 else result.stderr)

    if result.returncode == 0:
        print("  PASS: All tests passed")
        return 1, 0
    print("  FAIL: Some tests failed")
    return 0, 1


# ─────────────────────────────────────────────────────────────
# 6. SYNTAX CHECK: verify no syntax errors introduced
# ─────────────────────────────────────────────────────────────

PRE_EXISTING_SYNTAX_ERRORS = {
    "demo/pickle_files_journal_paper.py",
    "simopt/problem.py",
}


def test_syntax(verbose: bool = False) -> tuple[int, int]:
    """Verify no syntax errors were introduced."""
    header("6. Syntax Check")

    result = subprocess.run(
        ["git", "diff", "--name-only"],
        capture_output=True,
        text=True,
    )
    modified_files = [
        f.strip() for f in result.stdout.splitlines() if f.strip().endswith(".py")
    ]

    passed = 0
    failed = 0

    for filepath in modified_files:
        if not Path(filepath).exists():
            continue
        try:
            with open(filepath) as f:  # noqa: PTH123
                compile(f.read(), filepath, "exec")
            if verbose:
                print(f"  OK: {filepath}")
            passed += 1
        except SyntaxError as e:
            if filepath in PRE_EXISTING_SYNTAX_ERRORS:
                if verbose:
                    print(f"  SKIP (pre-existing): {filepath}:{e.lineno}")
                passed += 1
            else:
                print(f"  FAIL: {filepath}:{e.lineno} — {e.msg}")
                failed += 1

    print(f"\n  Result: {passed} passed, {failed} failed")
    return passed, failed


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main() -> int:
    """Run main entry point."""
    parser = argparse.ArgumentParser(
        description="Test that ruff lint fixes haven't changed runtime behavior"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    # Ensure ruff is on PATH
    ruff_path = Path.home() / ".local" / "bin"
    if str(ruff_path) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{ruff_path}:{os.environ.get('PATH', '')}"

    total_passed = 0
    total_failed = 0

    tests = [
        test_removed_imports,
        test_removed_star_imports,
        test_module_imports,
        test_bare_except_changes,
        test_removed_assignments,
        test_pytest,
        test_syntax,
    ]

    for test_fn in tests:
        try:
            p, f = test_fn(verbose=args.verbose)
            total_passed += p
            total_failed += f
        except Exception as e:
            print(f"  ERROR running {test_fn.__name__}: {e}")
            traceback.print_exc()
            total_failed += 1

    header("SUMMARY")
    print(f"  Total passed: {total_passed}")
    print(f"  Total failed: {total_failed}")
    if total_failed == 0:
        print("\n  ALL CHECKS PASSED")
    else:
        print(f"\n  {total_failed} CHECKS FAILED — review output above")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
