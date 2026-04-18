"""Generate SLURM job files from a CSV file and submit them to the SLURM queue."""

import csv
import json
import os
import shutil
import subprocess
import sys
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


def create_slurm_files(json_path: str, csv_file_name: str) -> list[str]:  # noqa: ARG001, D417
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
    script_path = demo_dir / "ASTROMoRF_HyperParameterSearch.py"

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

            # prepare arguments string in the form --problem_to_test <problem_name> --solver BO --proble_dim <dim_size>  # noqa: E501
            # args = " ".join('"'+str(x)+'"' for x in row)
            solver_name = row[0]
            problem_name = row[1]
            dim_size = row[2]
            args = f" --solver {solver_name} --problem_to_test {problem_name} --problem_dim {dim_size} "  # noqa: E501

            job_name = f"hyperparameter_{solver_name}_on_{problem_name}"
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


def main() -> None:
    """Main function to generate experiment CSV, create SLURM files, and submit them."""
    json_config_name = sys.argv[1]
    csv_output_path = sys.argv[2]
    csv_file = generate_csv(json_config_name, csv_output_path)
    generated_files = create_slurm_files(json_config_name, csv_file)
    run_slurm_files(generated_files)


if __name__ == "__main__":
    main()
