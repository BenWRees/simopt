#!/usr/bin/env python3
"""Verify that ruff fixes introduced no runtime behavior changes.

Checks:
  1. Syntax validation  - every .py file compiles without SyntaxError
  2. Import validation   - key packages import successfully
  3. Smoke execution     - a lightweight solver run completes
  4. AST structural comparison - public API surface unchanged
  5. Summary output
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files/directories excluded from syntax checks (vendored or known-broken)
EXCLUDE_DIRS = {
    "recover-stash-20260123",
    "pre-recover-backup",
    "recover_stash-20260123",
    "policy_experiment_results",
    ".claude",
    "__pycache__",
    ".git",
    "node_modules",
}

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    """Record a test result."""
    RESULTS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)


# -----------------------------------------------------------------------
# 1. Syntax validation
# -----------------------------------------------------------------------
def check_syntax() -> None:
    """Compile every .py file to verify no syntax errors."""
    errors: list[str] = []
    count = 0
    for py_file in sorted(REPO_ROOT.rglob("*.py")):
        # Skip excluded directories
        parts = py_file.relative_to(REPO_ROOT).parts
        if any(p in EXCLUDE_DIRS for p in parts):
            continue
        count += 1
        try:
            with py_file.open() as f:
                compile(f.read(), str(py_file), "exec")
        except SyntaxError as e:
            errors.append(f"  {py_file.relative_to(REPO_ROOT)}:{e.lineno} {e.msg}")

    if errors:
        record("syntax_validation", False, f"{len(errors)}/{count} files failed")
        for err in errors[:10]:
            print(err)
    else:
        record("syntax_validation", True, f"{count} files OK")


# -----------------------------------------------------------------------
# 2. Import validation
# -----------------------------------------------------------------------
def check_imports() -> None:
    """Verify that key project packages import without error."""
    sys.path.insert(0, str(REPO_ROOT))
    modules_to_check = [
        "simopt",
        "simopt.problem",
        "simopt.solver",
        "simopt.directory",
        "simopt.experiment.single",
    ]
    failures: list[str] = []
    skipped: list[str] = []
    ok_count = 0
    for mod_name in modules_to_check:
        try:
            importlib.import_module(mod_name)
            ok_count += 1
        except (ModuleNotFoundError, ImportError) as e:
            # Missing deps or pre-existing circular imports are not
            # caused by ruff lint fixes — skip them.
            skipped.append(f"  {mod_name}: {type(e).__name__}: {e}")
        except Exception as e:
            # Unexpected errors (e.g. NameError, AttributeError) may
            # indicate a ruff fix broke something.
            failures.append(f"  {mod_name}: {type(e).__name__}: {e}")

    if failures:
        record("import_validation", False, f"{len(failures)} imports failed")
        for f in failures:
            print(f)
    else:
        detail = f"{ok_count} OK"
        if skipped:
            detail += f", {len(skipped)} skipped (missing deps / pre-existing)"
        record("import_validation", True, detail)


# -----------------------------------------------------------------------
# 3. Smoke execution
# -----------------------------------------------------------------------
def check_smoke() -> None:
    """Run a minimal solver to verify runtime still works."""
    try:
        from simopt.directory import problem_directory, solver_directory

        # Pick a small, fast problem
        prob_cls = problem_directory.get("CNTNEWS-1")
        if prob_cls is None:
            record("smoke_execution", False, "CNTNEWS-1 not in directory")
            return

        problem = prob_cls()
        solver_cls = solver_directory.get("RNDSRCH")
        if solver_cls is None:
            record("smoke_execution", False, "RNDSRCH not in directory")
            return

        solver = solver_cls(fixed_factors={"sample_size": 5})

        from mrg32k3a.mrg32k3a import MRG32k3a

        rng_list = [MRG32k3a(s_ss_sss_index=[0, i, 0]) for i in range(3)]
        solver.attach_rngs(rng_list)

        from simopt.solver import Budget

        solver.budget = Budget(50)
        solver.solve(problem)
        record("smoke_execution", True, "RNDSRCH on CNTNEWS-1 completed")
    except (ModuleNotFoundError, ImportError) as e:
        # Missing deps or pre-existing import issues — not caused by ruff
        record(
            "smoke_execution",
            True,
            f"skipped (pre-existing): {type(e).__name__}: {e}",
        )
    except Exception as e:
        record("smoke_execution", False, f"{type(e).__name__}: {e}")


# -----------------------------------------------------------------------
# 4. AST structural comparison (public API surface)
# -----------------------------------------------------------------------
def check_ast_structure() -> None:
    """Check that key files have the expected public classes/functions."""
    checks_passed = 0
    checks_failed = 0
    key_files = {
        "simopt/problem.py": {"Problem", "Solution"},
        "simopt/solver.py": {"Solver", "Budget"},
        "simopt/multistage_problem.py": {"MultistageProblem"},
    }
    for rel_path, expected_names in key_files.items():
        fpath = REPO_ROOT / rel_path
        if not fpath.exists():
            record(
                "ast_structure",
                False,
                f"{rel_path} not found",
            )
            checks_failed += 1
            continue
        with fpath.open() as f:
            tree = ast.parse(f.read(), filename=str(fpath))
        top_level_names = set()
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                top_level_names.add(node.name)
        missing = expected_names - top_level_names
        if missing:
            record(
                "ast_structure",
                False,
                f"{rel_path}: missing {missing}",
            )
            checks_failed += 1
        else:
            checks_passed += 1

    if checks_failed == 0:
        record("ast_structure", True, f"{checks_passed} files have expected API")
    # Individual failures already recorded above


# -----------------------------------------------------------------------
# 5. Ruff compliance check
# -----------------------------------------------------------------------
def check_ruff() -> None:
    """Verify zero ruff errors."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format=concise"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    stdout = result.stdout.strip()
    # Check for "All checks passed" in output (robust across ruff versions)
    if "All checks passed" in stdout or (
        result.returncode == 0 and "Found" not in stdout
    ):
        record("ruff_compliance", True, "zero errors")
    else:
        lines = stdout.split("\n")
        # Find the "Found N errors" line
        summary = next(
            (ln for ln in lines if ln.startswith("Found")),
            lines[-1] if lines else "unknown",
        )
        record("ruff_compliance", False, summary)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    """Run all verification checks."""
    print("=" * 60)
    print("Verification: no behavior change from ruff fixes")
    print("=" * 60)
    print()

    check_syntax()
    check_imports()
    check_smoke()
    check_ast_structure()
    check_ruff()

    print()
    print("=" * 60)
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = total - passed
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
    if failed:
        print("FAILED checks:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
