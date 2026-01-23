#!/bin/bash
# Master script to submit all factor experiment jobs
# Generated automatically

echo "Submitting factor experiment jobs..."
echo "=================================="


echo "Submitting subspace_dimensions_facsize..."
sbatch /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/subspace_dimensions_facsize.slurm

echo "Submitting subspace_dimensions_dynamnews..."
sbatch /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/subspace_dimensions_dynamnews.slurm

echo "Submitting subspace_dimensions_network..."
sbatch /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/subspace_dimensions_network.slurm

echo "Submitting poly_basis_facsize..."
sbatch /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/poly_basis_facsize.slurm

echo "Submitting poly_basis_dynamnews..."
sbatch /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/poly_basis_dynamnews.slurm

echo "Submitting poly_basis_network..."
sbatch /Users/benjaminrees/Desktop/simopt/HPC_code/generated_slurm_files/poly_basis_network.slurm

echo ""
echo "=================================="
echo "All jobs submitted. Use 'squeue -u $USER' to check status."
