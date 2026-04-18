"""Generate SLURM job files from a CSV file and submit them to the SLURM queue."""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path.cwd().parent))


def find_path_upwards(start: Path, target: str) -> Path | None:
    """Search upward from 'start' for a directory or file named 'target'."""
    current = start.resolve()
    for parent in [current, *list(current.parents)]:
        candidate = parent / target
        if candidate.exists():
            return candidate
    return None


def find_path_downwards(start: Path, target: str) -> Path | None:
    """Search downward from 'start' for a directory or file named 'target' (limited.

    depth).
    """
    for p in start.rglob(target):
        return p
    return None


def load_config_from_json(
    json_path: str,
) -> tuple[list[str], list[str], list[int], dict, list[int]]:
    """Load configuration from a JSON file.

    Args:
        json_path (str): Path to the JSON configuration file.

    Returns:
        tuple: A tuple containing lists of solver names, problem names, dimension sizes,
        fixed factors, and budgets.
    """
    with open(json_path) as f:  # noqa: PTH123
        config = json.load(f)

    # Extract individual variables if needed
    solver_names = config.get("solver_names", [])
    problem_names = config.get("problem_names", [])
    dim_sizes = config.get("dim_sizes", [])
    fixed_factors = config.get("fixed_factors", {})
    budgets = config.get("budgets", [])

    return solver_names, problem_names, dim_sizes, fixed_factors, budgets


def generate_csv(json_path: str, csv_file_path: str) -> str:
    """Generate a CSV file for the arguments to pass to the slurm file.

    each row should be SOLVER NAME, PROBLEM NAME, PROBLEM DIMENSION, SOLVER_FACTORS,
    BUDGET, MACROREPLICATION NO.

    Args:
        json_path (str): Path to the JSON configuration file.
        csv_file_path (str): Path to save the generated CSV file.

    Returns:
        str: Path to the generated CSV file.
    """
    # === CONFIGURATION ===
    csv_file = csv_file_path  # your CSV file

    # === NAMES ===
    (
        solver_names,
        problem_names,
        dim_sizes,
        fixed_factors,
        budgets,
    ) = load_config_from_json(json_path)

    # === CREATE CSV FILE ===
    with open(csv_file, mode="w", newline="") as file:  # noqa: PTH123
        writer = csv.writer(file)
        # Write header
        writer.writerow(
            [
                "solver_name",
                "problem_name",
                "dim_size",
                "solver_factors",
                "budget",
                "macroreplication_no",
            ]
        )

        # Generate combinations and write to CSV
        for solver, fixed_factor in zip(solver_names, fixed_factors, strict=False):
            for problem in problem_names:
                for dim in dim_sizes:
                    for budget in budgets:
                        writer.writerow(
                            [solver, problem, dim, str(fixed_factor), budget, 100]
                        )
            writer.writerow([])  # Blank line between different problems

    print(f"✅ Generated CSV file '{csv_file}' with experiment setups.")
    return csv_file


def create_slurm_files(  # noqa: D417
    json_path: str,  # noqa: ARG001
    csv_file_name: str,
    base_output_dir: str | None = None,
) -> list[str]:
    """Create SLURM job files based on a CSV file and a base SLURM template.

    Args:
        csv_file_name (str): Path to the CSV file containing experiment setups.

    Returns:
        list[str]: List of generated SLURM job file paths.
    """
    # === CONFIGURATION ===
    csv_file = csv_file_name  # your CSV file

    start_dir = Path.cwd().resolve()
    print(f"Starting search from: {start_dir}")

    # Try to locate 'simopt' directory upward or downward
    simopt_dir = find_path_upwards(start_dir, "simopt") or find_path_downwards(
        start_dir, "simopt"
    )
    if not simopt_dir:
        raise FileNotFoundError("Could not find 'simopt' directory from current path.")

    print(f"Found simopt directory: {simopt_dir}")

    # Define subdirectories relative to simopt
    hpc_code_dir = simopt_dir / "HPC_code"
    demo_dir = simopt_dir / "demo"

    # Define files
    base_slurm_file = hpc_code_dir / "journal_setup.slurm"
    script_path = demo_dir / "pickle_OMoRF.py"

    # Define (and create) generated folder
    output_dir = hpc_code_dir / "generated_slurm_files"
    output_dir.mkdir(exist_ok=True)

    # === READ CSV ===
    with open(csv_file, newline="") as f:  # noqa: PTH123
        reader = csv.reader(f)
        next(reader, None)  # skip header if present

        generated_files = []

        for _i, row in enumerate(reader, start=1):
            if row == []:
                continue  # skip blank lines

            args = " ".join('"' + str(x) + '"' for x in row)
            solver_name = row[0]
            problem_name = row[1]
            job_name = f"journal_setup_{solver_name}_on_{problem_name}"
            new_filename = os.path.join(output_dir, f"{job_name}.slurm")  # noqa: PTH118

            # Copy the base SLURM template
            shutil.copy(base_slurm_file, new_filename)

            # === Update job-name line ===
            with open(new_filename) as file:  # noqa: PTH123
                content = file.readlines()

            for j, line in enumerate(content):
                if line.strip().startswith("#SBATCH --job-name="):
                    content[j] = f"#SBATCH --job-name={job_name}\n"
                    break  # stop after finding the first match

            # Write back the modified content
            with open(new_filename, "w") as file:  # noqa: PTH123
                file.writelines(content)

            # === Append the run command ===
            with open(new_filename, "a") as out:  # noqa: PTH123
                if base_output_dir:
                    outdir = Path(base_output_dir) / job_name
                    out.write(
                        f'\npython {script_path} {args} --output-dir "{outdir}"\n'
                    )
                else:
                    out.write(f"\npython {script_path} {args}\n")

            generated_files.append(new_filename)

    print(f"✅ Generated {len(generated_files)} SLURM job files in '{output_dir}/'")

    return generated_files


def run_slurm_files(generated_files: list[str]) -> None:
    """Submit SLURM job files to the queue.

    Args:
        generated_files (list[str]): List of SLURM job file paths to submit.
    """
    # === SUBMIT EACH JOB TO THE QUEUE ===
    print("\n🚀 Submitting jobs to SLURM queue...")
    for file in generated_files:
        try:
            result = subprocess.run(
                ["sbatch", file], check=True, capture_output=True, text=True
            )
            print(f"Submitted {os.path.basename(file)} → {result.stdout.strip()}")  # noqa: PTH119
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to submit {file}: {e.stderr.strip()}")

    print("\n✅ All jobs processed.")


def run_slurm_files_with_resubmit_on_timeout(
    generated_files: list[str],
    poll_interval: int = 60,
    max_resubmissions: int = 0,
    verbose: bool = True,
) -> None:
    """Submit jobs, monitor them, and resubmit any that TIMEOUT.

    Args:
        generated_files: list of slurm file paths to submit
        poll_interval: seconds between polls
        max_resubmissions: 0 means unlimited
        verbose: print status messages
    """
    # Submit and capture job IDs
    file_map: dict[str, dict] = {}

    for file in generated_files:
        try:
            result = subprocess.run(
                ["sbatch", file], check=True, capture_output=True, text=True
            )
            out = result.stdout.strip()
            # sbatch typically prints: Submitted batch job 12345
            m = re.search(r"(\d+)", out)
            if m:
                jobid = m.group(1)
                file_map[file] = {"jobid": jobid, "resubmissions": 0}
                if verbose:
                    print(f"Submitted {os.path.basename(file)} → job {jobid}")  # noqa: PTH119
            else:
                print(
                    f"Could not parse job id from sbatch output: '{out}' for file {file}"  # noqa: E501
                )
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit {file}: {e.stderr.strip()}")

    if verbose:
        print(
            "\nMonitoring submitted jobs for TIMEOUTs; will resubmit only TIMEOUTed jobs."  # noqa: E501
        )

    # Monitor loop
    while file_map:
        for file, info in list(file_map.items()):
            jobid = info["jobid"]

            # If job still in squeue, skip
            try:
                squeue = subprocess.run(
                    ["squeue", "-j", str(jobid), "-h"], capture_output=True, text=True
                )
                if squeue.stdout.strip():
                    # still running/pending
                    if verbose:
                        print(
                            f"Job {jobid} for {os.path.basename(file)} still in queue."  # noqa: PTH119
                        )
                    continue
            except Exception:
                # If squeue fails, we'll fall back to sacct
                pass

            # Job no longer in squeue; inspect final state via sacct
            state = None
            exitcode = None
            try:
                sacct = subprocess.run(
                    ["sacct", "-j", str(jobid), "-o", "State,ExitCode", "-n", "-P"],
                    capture_output=True,
                    text=True,
                )
                sacct_out = sacct.stdout.strip()
                if sacct_out:
                    # sacct may return multiple lines; take the first non-empty
                    first_line = sacct_out.splitlines()[0]
                    parts = first_line.split("|")
                    if len(parts) >= 1:
                        state = parts[0]
                    if len(parts) >= 2:
                        exitcode = parts[1]
            except Exception:
                pass

            # Fallback to scontrol if sacct didn't give useful info
            if not state:
                try:
                    sctrl = subprocess.run(
                        ["scontrol", "show", "job", str(jobid)],
                        capture_output=True,
                        text=True,
                    )
                    sctrl_out = sctrl.stdout
                    mstate = re.search(r"JobState=(\w+)", sctrl_out)
                    if mstate:
                        state = mstate.group(1)
                except Exception:
                    pass

            if not state:
                print(
                    f"Could not determine final state for job {jobid} (file {file}); removing from watch list."  # noqa: E501
                )
                file_map.pop(file, None)
                continue

            if verbose:
                print(
                    f"Job {jobid} (file {os.path.basename(file)}) finished with state='{state}', exitcode='{exitcode}'"  # noqa: E501, PTH119
                )

            # If TIMEOUT, resubmit (unless max resubmissions reached)
            if "TIMEOUT" in state.upper():
                info["resubmissions"] += 1
                if max_resubmissions and info["resubmissions"] > max_resubmissions:
                    print(
                        f"Max resubmissions reached for {file}; not resubmitting further."  # noqa: E501
                    )
                    file_map.pop(file, None)
                    continue

                print(f"Resubmitting {file} due to TIMEOUT (previous job {jobid}).")
                try:
                    res = subprocess.run(
                        ["sbatch", file], check=True, capture_output=True, text=True
                    )
                    m = re.search(r"(\d+)", res.stdout.strip())
                    if m:
                        new_jobid = m.group(1)
                        info["jobid"] = new_jobid
                        if verbose:
                            print(
                                f"Resubmitted {os.path.basename(file)} → job {new_jobid} (resubmissions={info['resubmissions']})"  # noqa: E501, PTH119
                            )
                        # continue watching
                        continue
                    print(
                        f"Could not parse job id from resubmit output: '{res.stdout.strip()}'"  # noqa: E501
                    )
                    file_map.pop(file, None)
                    continue
                except subprocess.CalledProcessError as e:
                    print(f"Failed to resubmit {file}: {e.stderr.strip()}")
                    file_map.pop(file, None)
                    continue

            # If completed or cancelled, do not resubmit
            if any(
                x in state.upper()
                for x in ("COMPLETED", "CANCELLED", "CANCELLED+", "CANCELLED")
            ):
                print(
                    f"Job {jobid} for {os.path.basename(file)} completed with state={state}; not resubmitting."  # noqa: E501, PTH119
                )
                file_map.pop(file, None)
                continue

            # For other terminal states (FAILED, NODE_FAIL, etc.), do not resubmit by default  # noqa: E501
            if any(
                x in state.upper()
                for x in ("FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED")
            ):
                print(f"Job {jobid} ended with state={state}; not resubmitting.")
                file_map.pop(file, None)
                continue

            # Unknown state: remove from watch list to avoid infinite loops
            print(
                f"Job {jobid} ended with unhandled state='{state}'; not resubmitting."
            )
            file_map.pop(file, None)

        # Sleep before next poll if there are still jobs
        if file_map:
            time.sleep(poll_interval)

    print("\n✅ Monitoring complete; no jobs remaining to watch.")


def main() -> None:
    """Main function to generate experiment CSV, create SLURM files, and submit them."""
    if len(sys.argv) < 3:
        print(
            "Usage: generate_slurm_files_omorf.py <json_config> <csv_output> [--resubmit-timeout]"  # noqa: E501
        )
        sys.exit(1)

    json_config_name = sys.argv[1]
    csv_output_path = sys.argv[2]
    resubmit_flag = "--resubmit-timeout" in sys.argv[3:]

    csv_file = generate_csv(json_config_name, csv_output_path)
    base_output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    generated_files = create_slurm_files(json_config_name, csv_file, base_output_dir)

    if resubmit_flag:
        poll_interval = int(os.environ.get("RESUBMIT_POLL", "60"))
        max_resubmissions = int(os.environ.get("RESUBMIT_MAX", "0"))
        run_slurm_files_with_resubmit_on_timeout(
            generated_files,
            poll_interval=poll_interval,
            max_resubmissions=max_resubmissions,
        )
    else:
        run_slurm_files(generated_files)


if __name__ == "__main__":
    main()
