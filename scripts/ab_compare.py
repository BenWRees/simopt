#!/usr/bin/env python3
"""
A/B comparison driver for ASTROMoRF adaptive-dimension safety rules.

Creates two temporary module copies:
 - relaxed: uses the current `simopt/solvers/astromorf.py` (already in workspace)
 - conservative: patches the safety block to the original conservative settings

Runs `run_solver_test.py` repeatedly for each arm, parses the final objective
from the printed "Iteration DataFrame" section, and reports median final
objective and saves simple convergence CSVs under `ab_results/`.

Usage:
    python scripts/ab_compare.py --repeats 10 --budget 1000 --problem rosenbrock

Note: run from workspace root. Requires your usual Python env (conda). The
script will not install dependencies.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_MODULE = ROOT / "simopt" / "solvers" / "astromorf.py"
RUN_SCRIPT = ROOT / "run_solver_test.py"
RESULT_DIR = ROOT / "ab_results"
RESULT_DIR.mkdir(exist_ok=True)

RELAXED_SNIPPET = r"""
        # Prevent extreme oscillations: require a small cooldown before applying changes
        # (Relaxed rules: cooldown is short and step-cap is proportional to problem dimension)
        if not self.in_dimension_reset:
            time_since_change = self.iteration_count - getattr(self, "last_d_change_iteration", 0)
            if time_since_change < min_iters_between_d_changes:
                # Avoid changing d too frequently within the tiny cooldown window
                return self.d

            # Relaxed step cap: allow larger jumps proportional to problem dimension
            # Default to 90% of the problem dimension (but at least 3)
            max_step = max(3, int(0.9 * max(1, self.problem.dim)))

            # If multiple consecutive unsuccessful iterations, allow full-range jump
            if getattr(self, "consecutive_unsuccessful", 0) >= 3:
                max_step = max(max_step, max(1, self.problem.dim - 1))

            if abs(optimal_d - self.d) > max_step:
                optimal_d = int(self.d + np.sign(optimal_d - self.d) * max_step)
"""

CONSERVATIVE_SNIPPET = r"""
        # Prevent extreme oscillations: require a small cooldown before applying changes
        if not self.in_dimension_reset:
            time_since_change = self.iteration_count - getattr(self, "last_d_change_iteration", 0)
            if time_since_change < min_iters_between_d_changes:
                # Avoid changing d too frequently
                return self.d

            # Conservative step cap to avoid large jumps
            max_step = 2

            # If multiple consecutive unsuccessful iterations, be slightly more willing to increase d
            if getattr(self, "consecutive_unsuccessful", 0) >= 3:
                max_step = max(max_step, 3)

            # Cap the step to a reasonable fraction of problem dimension
            cap = max(1, int(0.5 * max(1, self.problem.dim)))
            max_step = min(max_step, cap)

            if abs(optimal_d - self.d) > max_step:
                optimal_d = int(self.d + np.sign(optimal_d - self.d) * max_step)
"""


def make_patched_module(temp_root: Path, conservative: bool) -> None:
    """Write a patched `simopt/solvers/astromorf.py` under temp_root.

    If conservative=True, replace the relaxed snippet with the conservative one.
    Otherwise copy the current file as-is.
    """
    tgt = temp_root / "simopt" / "solvers"
    tgt.mkdir(parents=True, exist_ok=True)
    text = SRC_MODULE.read_text()
    if conservative:
        if RELAXED_SNIPPET in text:
            text = text.replace(RELAXED_SNIPPET, CONSERVATIVE_SNIPPET)
        else:
            # Try a coarse replacement of the relaxed comment header to the conservative block
            text = text.replace("# (Relaxed rules: cooldown is short and step-cap is proportional to problem dimension)", "")
            # Best-effort: insert conservative_snippet after the `if not self.in_dimension_reset:` anchor
            anchor = "if not self.in_dimension_reset:"
            if anchor in text:
                parts = text.split(anchor, 1)
                # Find end of that if-block by searching next blank line after anchor occurrence
                text = parts[0] + anchor + "\n" + CONSERVATIVE_SNIPPET + parts[1]
    # else: use existing current file content
    (tgt / "astromorf.py").write_text(text)


def run_one(temp_path: Path, problem: str, budget: int) -> float:
    env = os.environ.copy()
    # Ensure our temp_path is searched before workspace so our patched module is imported
    env["PYTHONPATH"] = str(temp_path) + os.pathsep + str(ROOT)
    # Run the run_solver_test.py script and capture stdout
    proc = subprocess.run(
        [sys.executable, str(RUN_SCRIPT), problem],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        cwd=str(ROOT),
    )
    out = proc.stdout
    # parse final fn_estimate from the printed Iteration DataFrame table: find last numeric row
    # fallback: search for 'fn_estimate' header and then parse subsequent lines
    if "Iteration DataFrame:" in out:
        after = out.split("Iteration DataFrame:", 1)[1]
        lines = [ln.strip() for ln in after.splitlines() if ln.strip()]
        # lines include header row then rows; find last row that looks like numbers
        last_val = None
        for ln in reversed(lines):
            cols = ln.split()
            # expect columns: iteration budget_history fn_estimate mrep
            if len(cols) >= 3:
                try:
                    val = float(cols[-2]) if cols[-1].isdigit() else float(cols[-2])
                    last_val = val
                    break
                except Exception:
                    continue
        if last_val is not None:
            return last_val
    # fallback: try to find 'fn_estimate' occurrences and take numeric following
    import re

    matches = re.findall(r"fn_estimate\s*\n(.*?)\n", out, re.S)
    if matches:
        try:
            rows = [ln for ln in matches[0].splitlines() if ln.strip()]
            if rows:
                last = rows[-1].split()
                return float(last[-2])
        except Exception:
            pass

    # As ultimate fallback, attempt to parse any floating numbers and return last
    floats = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", out)]
    return floats[-1] if floats else float("nan")


def run_arm(arm_name: str, conservative: bool, repeats: int, problem: str, budget: int):
    """Run `repeats` trials for the arm and return list of final objectives."""
    vals = []
    csv_lines = []
    for i in range(repeats):
        with tempfile.TemporaryDirectory() as td:
            tdpath = Path(td)
            # copy minimal package structure and patched module
            make_patched_module(tdpath, conservative=conservative)
            # copy other package files that may be required by imports? We rely on PYTHONPATH ordering
            val = run_one(tdpath, problem, budget)
            print(f"{arm_name} run {i+1}/{repeats}: final_obj={val}")
            vals.append(val)
            csv_lines.append(f"{i+1},{val}")
    # save per-arm CSV
    out_file = RESULT_DIR / f"{arm_name}_results.csv"
    out_file.write_text("run,final_obj\n" + "\n".join(csv_lines))
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--problem", type=str, default="rosenbrock")
    args = parser.parse_args()

    repeats = args.repeats
    budget = args.budget
    problem = args.problem

    print("Running A/B comparison:")
    print(f"  repeats={repeats}, budget={budget}, problem={problem}")

    # Arm A: relaxed (current file)
    relaxed_vals = run_arm("relaxed", conservative=False, repeats=repeats, problem=problem, budget=budget)
    # Arm B: conservative (patched)
    conservative_vals = run_arm("conservative", conservative=True, repeats=repeats, problem=problem, budget=budget)

    print("\nSummary:")
    print(f" relaxed: median={statistics.median(relaxed_vals):.6g}, mean={statistics.mean(relaxed_vals):.6g}")
    print(f" conservative: median={statistics.median(conservative_vals):.6g}, mean={statistics.mean(conservative_vals):.6g}")
    print(f"Detailed CSVs in: {RESULT_DIR}")


if __name__ == '__main__':
    main()
