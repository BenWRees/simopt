#!/bin/bash
# Master script to submit all factor experiment jobs
# Generated automatically

echo "Submitting factor experiment jobs..."
echo "=================================="


echo "Submitting subspace_dimensions_san..."
sbatch /Users/benjaminrees/Desktop/simopt/factor_experiments/subspace_dimensions_san.slurm

echo "Submitting subspace_dimensions_dynamnews..."
sbatch /Users/benjaminrees/Desktop/simopt/factor_experiments/subspace_dimensions_dynamnews.slurm

echo "Submitting subspace_dimensions_network..."
sbatch /Users/benjaminrees/Desktop/simopt/factor_experiments/subspace_dimensions_network.slurm

echo "Submitting subspace_dimensions_rosenbrock..."
sbatch /Users/benjaminrees/Desktop/simopt/factor_experiments/subspace_dimensions_rosenbrock.slurm

echo "Submitting poly_basis_san..."
sbatch /Users/benjaminrees/Desktop/simopt/factor_experiments/poly_basis_san.slurm

echo "Submitting poly_basis_dynamnews..."
sbatch /Users/benjaminrees/Desktop/simopt/factor_experiments/poly_basis_dynamnews.slurm

echo "Submitting poly_basis_network..."
sbatch /Users/benjaminrees/Desktop/simopt/factor_experiments/poly_basis_network.slurm

echo "Submitting poly_basis_rosenbrock..."
sbatch /Users/benjaminrees/Desktop/simopt/factor_experiments/poly_basis_rosenbrock.slurm

echo ""
echo "=================================="
echo "All jobs submitted. Use 'squeue -u $USER' to check status."
