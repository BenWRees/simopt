#!/usr/bin/env python3
"""Generate SLURM job files for factor experiments from a JSON configuration.

This script reads a configuration file and generates SLURM array job scripts
for running ASTROMoRF factor experiments (subspace dimensions or polynomial basis types)
on an HPC cluster.

Usage:
    python generate_factor_experiment_jobs.py factor_experiments_config.json

    # Submit all generated jobs
    python generate_factor_experiment_jobs.py factor_experiments_config.json --submit
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path) as f:  # noqa: PTH123
        return json.load(f)


def calculate_n_tasks(experiment: dict) -> int:
    """Calculate the number of tasks (design points) for an experiment."""
    if experiment["factor_type"] == "subspace":
        # Number of subspace dimensions to test
        return len(experiment.get("subspace_dims", list(range(1, 9))))
    if experiment["factor_type"] == "basis":
        # Number of basis types to test
        default_bases = [
            "hermite",
            "legendre",
            "chebyshev",
            "monomial",
            "natural",
            "laguerre",
            "nfp",
            "lagrange",
            "monomial_poly",
        ]
        return len(experiment.get("basis_types", default_bases))
    if experiment["factor_type"] == "full":
        # Full factorial: subspace dims × basis types  # noqa: RUF003
        n_dims = len(experiment.get("subspace_dims", list(range(1, 9))))
        default_bases = [
            "hermite",
            "legendre",
            "chebyshev",
            "monomial",
            "natural",
            "laguerre",
            "nfp",
            "lagrange",
            "monomial_poly",
        ]
        n_bases = len(experiment.get("basis_types", default_bases))
        return n_dims * n_bases
    raise ValueError(f"Unknown factor type: {experiment['factor_type']}")


def generate_slurm_script(
    experiment: dict,
    slurm_settings: dict,
    output_dir: Path,
    demo_script_path: Path,
) -> Path:
    """Generate a SLURM array job script for an experiment."""
    n_tasks = calculate_n_tasks(experiment)
    exp_name = experiment["name"]

    # Build command arguments
    cmd_args = [
        f"--factor {experiment['factor_type']}",
        f"--problem {experiment['problem_name']}",
        f"--dim {experiment['problem_dim']}",
        f"--budget {experiment['budget']}",
        f"--n-macroreps {experiment['n_macroreps']}",
        f"--n-postreps {experiment['n_postreps']}",
        f"--polynomial-degree {experiment.get('polynomial_degree', 2)}",
        f"--output-dir {output_dir / exp_name}",
        # "--task-id $SLURM_ARRAY_TASK_ID",
    ]

    # Add factor-specific arguments
    if experiment["factor_type"] == "subspace":
        if "subspace_dims" in experiment:
            dims_str = ",".join(str(d) for d in experiment["subspace_dims"])
            cmd_args.append(f"--subspace-dims {dims_str}")

        if "fixed_basis" in experiment:
            # This would be the default basis when testing subspace dims
            pass  # The script handles this with defaults

    elif experiment["factor_type"] == "basis":
        if "basis_types" in experiment:
            bases_str = ",".join(experiment["basis_types"])
            cmd_args.append(f"--basis-types {bases_str}")
        if "fixed_subspace_dim" in experiment:
            cmd_args.append(f"--fixed-subspace-dim {experiment['fixed_subspace_dim']}")

    # Create log directory
    log_dir = output_dir / exp_name / "logs"

    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={exp_name}
#SBATCH --partition={slurm_settings.get("partition", "batch")}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_settings.get("cpus_per_task", 20)}
#SBATCH --mem-per-cpu={slurm_settings.get("mem_per_cpu", "8G")}
#SBATCH --time={slurm_settings.get("time_limit", "60:00:00")}
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --output={log_dir}/slurm_%A_%a.out
#SBATCH --error={log_dir}/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user={slurm_settings.get("mail_user", "$USER@soton.ac.uk")}

# ============================================================================
# SLURM Array Job Script for ASTROMoRF Factor Experiments
# Experiment: {exp_name}
# Factor Type: {experiment["factor_type"]}
# Problem: {experiment["problem_name"]}
# Number of Tasks: {n_tasks}
# ============================================================================

echo "Starting task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_JOB_ID"
echo "Running on host: $(hostname)"
echo "Time: $(date)"

# Create output and log directories
mkdir -p {output_dir / exp_name}
mkdir -p {log_dir}

# Activate conda environment
source $HOME/miniconda3/bin/activate simopt

# Navigate to simopt directory
#cd {demo_script_path.parent.parent}

# Run the experiment for this task ID
python {demo_script_path} {" ".join(cmd_args)} --task-id $SLURM_ARRAY_TASK_ID
--generate_csv

exit_code=$?

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $exit_code"
echo "End time: $(date)"

exit $exit_code"""

    # Write SLURM script
    slurm_file = output_dir / f"{exp_name}.slurm"
    slurm_file.parent.mkdir(parents=True, exist_ok=True)

    with open(slurm_file, "w") as f:  # noqa: PTH123
        f.write(slurm_content)

    # Make executable
    os.chmod(slurm_file, 0o755)  # noqa: PTH101

    return slurm_file


def generate_master_submit_script(
    slurm_files: list[Path],
    output_dir: Path,
) -> Path:
    """Generate a master script to submit all SLURM jobs."""
    script_content = """#!/bin/bash
# Master script to submit all factor experiment jobs
# Generated automatically

echo "Submitting factor experiment jobs..."
echo "=================================="

"""

    for slurm_file in slurm_files:
        script_content += f"""
echo "Submitting {slurm_file.stem}..."
sbatch {slurm_file}
"""

    script_content += """
echo ""
echo "=================================="
echo "All jobs submitted. Use 'squeue -u $USER' to check status."
"""

    submit_script = output_dir / "submit_all_experiments.sh"
    with open(submit_script, "w") as f:  # noqa: PTH123
        f.write(script_content)

    os.chmod(submit_script, 0o755)  # noqa: PTH101

    return submit_script


def generate_results_collector_script(
    experiments: list[dict],
    output_dir: Path,
) -> Path:
    """Generate a Python script to collect and analyze results after experiments.

    complete.
    """
    script_content = (
        '''#!/usr/bin/env python3
"""
Collect and analyze results from factor experiments.

This script reads the pickle files generated by the factor experiments
and produces summary statistics and visualizations.
"""

import pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def collect_results(experiment_dir: Path) -> pd.DataFrame:
    """Collect all results from an experiment directory."""
    results = []
    
    for pickle_file in experiment_dir.glob("*.pickle"):
        try:
            with open(pickle_file, "rb") as f:
                data = pickle.load(f)
            
            # Extract relevant information
            # (Structure depends on ProblemSolver output format)
            result = {
                "file": pickle_file.name,
                # Add more fields based on actual pickle structure
            }
            results.append(result)
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
    
    return pd.DataFrame(results)


def main():
    output_dir = Path("'''
        + str(output_dir)
        + """")
    
    # List of experiment directories
    experiment_names = """
        + str([exp["name"] for exp in experiments])
        + """
    
    for exp_name in experiment_names:
        exp_dir = output_dir / exp_name
        if exp_dir.exists():
            print(f"\\nCollecting results for {exp_name}...")
            df = collect_results(exp_dir)
            print(f"  Found {len(df)} result files")
            
            # Save summary
            summary_file = exp_dir / "results_summary.csv"
            df.to_csv(summary_file, index=False)
            print(f"  Saved summary to {summary_file}")
        else:
            print(f"\\nDirectory not found: {exp_dir}")


if __name__ == "__main__":
    main()
"""
    )

    collector_script = output_dir / "collect_results.py"
    with open(collector_script, "w") as f:  # noqa: PTH123
        f.write(script_content)

    os.chmod(collector_script, 0o755)  # noqa: PTH101

    return collector_script


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM job files for factor experiments",
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="factor_experiments",
        help="Output directory for SLURM scripts and results",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs after generating scripts",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)
    experiments = config.get("experiments", [])
    slurm_settings = config.get("slurm_settings", {})

    if not experiments:
        print("No experiments defined in configuration file.")
        return

    # Setup paths
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the journal_factors_test.py script
    hpc_dir = Path(__file__).parent
    demo_dir = hpc_dir.parent / "demo"
    demo_script = demo_dir / "journal_factors_test.py"

    if not demo_script.exists():
        print(f"Error: Could not find {demo_script}")
        return

    print(f"Generating SLURM scripts for {len(experiments)} experiments...")
    print(f"Output directory: {output_dir}")
    print(f"Demo script: {demo_script}")
    print()

    # Generate SLURM scripts for each experiment
    slurm_files = []
    for experiment in experiments:
        slurm_file = generate_slurm_script(
            experiment=experiment,
            slurm_settings=slurm_settings,
            output_dir=output_dir,
            demo_script_path=demo_script,
        )
        slurm_files.append(slurm_file)
        n_tasks = calculate_n_tasks(experiment)
        print(f"    -> {slurm_file.name} ({n_tasks} tasks)")

    # Generate master submit script
    submit_script = generate_master_submit_script(slurm_files, output_dir)
    print(f"\nGenerated master submit script: {submit_script}")

    # Generate results collector script
    collector_script = generate_results_collector_script(experiments, output_dir)
    print(f"Generated results collector: {collector_script}")

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Total SLURM scripts: {len(slurm_files)}")
    print(f"  Output directory: {output_dir}")
    print("\nTo submit all jobs, run:")
    print(f"  bash {submit_script}")
    print("\nOr submit individually:")
    for sf in slurm_files:
        print(f"  sbatch {sf}")

    # Optionally submit jobs
    if args.submit:
        print("\n" + "=" * 60)
        print("Submitting jobs...")
        for slurm_file in slurm_files:
            try:
                result = subprocess.run(
                    ["sbatch", str(slurm_file)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"  Submitted {slurm_file.name}: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  Failed to submit {slurm_file.name}: {e.stderr}")
            except FileNotFoundError:
                print("  Error: sbatch command not found (not on an HPC node?)")
                break


if __name__ == "__main__":
    main()
