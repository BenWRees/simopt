#!/bin/bash
#SBATCH --job-name=poly_basis_facsize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6:00:00
#SBATCH --array=0-6
#SBATCH --output=/Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/poly_basis_facsize/logs/slurm_%A_%a.out
#SBATCH --error=/Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/poly_basis_facsize/logs/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=B.Rees@soton.ac.uk

# ============================================================================
# SLURM Array Job Script for ASTROMoRF Factor Experiments
# Experiment: poly_basis_facsize
# Factor Type: basis
# Problem: FACSIZE-1
# Number of Tasks: 7
# ============================================================================

echo "Starting task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_JOB_ID"
echo "Running on host: $(hostname)"
echo "Time: $(date)"

# Create output and log directories
mkdir -p /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/poly_basis_facsize
mkdir -p /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/poly_basis_facsize/logs

# Activate conda environment
source $HOME/miniconda3/bin/activate simopt

# Navigate to simopt directory
cd /Users/benjaminrees/Desktop/simopt

# Run the experiment for this task ID
python /Users/benjaminrees/Desktop/simopt/demo/journal_factors_test.py \
    --factor basis --problem FACSIZE-1 --dim 100 --budget 50000 --n-macroreps 20 --n-postreps 100 --polynomial-degree 2 --output-dir /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/poly_basis_facsize --task-id $SLURM_ARRAY_TASK_ID --basis-types hermite,legendre,chebyshev,monomial,natural,laguerre,nfp \
    --task-id $SLURM_ARRAY_TASK_ID

exit_code=$?

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $exit_code"
echo "End time: $(date)"

exit $exit_code
