"""
    Generate SLURM job files from a CSV file and submit them to the SLURM queue.
"""

import csv
import os
import shutil
import json 
import subprocess
import sys
from typing import Dict, List
from pathlib import Path

# Take the current directory, find the parent, and add it to the system path
sys.path.append(str(Path.cwd().parent))

def load_config_from_json(json_path: str) -> tuple[list[str], list[str], list[int], dict, list[int]] :
    """
    Load configuration from a JSON file.

    Args:
        json_path (str): Path to the JSON configuration file.

    Returns:
        tuple: A tuple containing lists of solver names, problem names, dimension sizes, fixed factors, and budgets.
    """
    with open(json_path, 'r') as f:
        config = json.load(f)

    # Extract individual variables if needed
    solver_names = config.get("solver_names", [])
    problem_names = config.get("problem_names", [])
    dim_sizes = config.get("dim_sizes", [])
    fixed_factors = config.get("fixed_factors", {})
    budgets = config.get("budgets", [])

    return solver_names, problem_names, dim_sizes, fixed_factors, budgets

def generate_csv(json_path: str, csv_file_path: str) -> str :
    """
        Generate a CSV file with combinations of solvers, problems, dimensions, and budgets.

        Args:
            json_path (str): Path to the JSON configuration file.
            csv_file_path (str): Path to save the generated CSV file.

        Returns:
            str: Path to the generated CSV file.
    """
    # === CONFIGURATION ===
    csv_file = csv_file_path        # your CSV file


    # === NAMES ===
    solver_names, problem_names, dim_sizes, fixed_factors, budgets = load_config_from_json(json_path)

    # === CREATE CSV FILE ===
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['problem_name', 'dim_size', 'solver_factors', 'budget'])
        
        # Generate combinations and write to CSV
        for problem in problem_names:
            for dim in dim_sizes:
                for budget in budgets:
                    writer.writerow([problem, dim, str(fixed_factors), budget])
            writer.writerow([])  # Blank line between different problems

    print(f"✅ Generated CSV file '{csv_file}' with experiment setups.")
    return csv_file

def create_slurm_files(json_path: str, csv_file_name: str) -> list[str] : 
    """
        Create SLURM job files based on a CSV file and a base SLURM template.

        Args:
            csv_file_name (str): Path to the CSV file containing experiment setups.

        Returns:
            list[str]: List of generated SLURM job file paths.
    """
    # === CONFIGURATION ===
    csv_file = csv_file_name                    # your CSV file
    base_slurm_file = "journal_setup.slurm"         # your base SLURM template
    output_dir = "generated_slurm_files"                   # directory to store new SLURM files
    script_path = "simopt/demo/pickle_files_journal_paper.py"  # your Python script

    # === CREATE OUTPUT DIRECTORY ===
    os.makedirs(output_dir, exist_ok=True)

    # === READ CSV ===
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if present

        generated_files = []

        for i, row in enumerate(reader, start=1):
            if row == []:
                continue  # skip blank lines

            args = json_path + " " + " ".join('"'+str(x)+'"' for x in row)
            problem_name = row[0]
            job_name = f"journal_setup_experiment_on_{problem_name}"
            new_filename = os.path.join(output_dir, f"{job_name}.slurm")

            # Copy the base SLURM template
            shutil.copy(base_slurm_file, new_filename)

            # === Update job-name line ===
            with open(new_filename, "r") as file:
                content = file.readlines()

            for j, line in enumerate(content):
                if line.strip().startswith("#SBATCH --job-name="):
                    content[j] = f"#SBATCH --job-name={job_name}\n"
                    break  # stop after finding the first match

            # Write back the modified content
            with open(new_filename, "w") as file:
                file.writelines(content)

             # === Append the run command ===
            with open(new_filename, "a") as out:
                out.write(f"\npython {script_path} {args}\n")

            generated_files.append(new_filename)

    print(f"✅ Generated {len(generated_files)} SLURM job files in '{output_dir}/'")

    return generated_files

def run_slurm_files(generated_files: list[str]) -> None :
    """
        Submit SLURM job files to the queue.

        Args:
            generated_files (list[str]): List of SLURM job file paths to submit.
    """
    # === SUBMIT EACH JOB TO THE QUEUE ===
    print("\n🚀 Submitting jobs to SLURM queue...")
    for file in generated_files:
        try:
            result = subprocess.run(["sbatch", file], check=True, capture_output=True, text=True)
            print(f"Submitted {os.path.basename(file)} → {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to submit {file}: {e.stderr.strip()}")

    print("\n✅ All jobs processed.")


def main() : 
    """
        Main function to generate experiment CSV, create SLURM files, and submit them.
    """
    json_config_name = sys.argv[1]
    csv_output_path = sys.argv[2]
    csv_file = generate_csv(json_config_name, csv_output_path)
    generated_files = create_slurm_files(json_config_name, csv_file)
    run_slurm_files(generated_files)

if __name__ == "__main__":
    main()