#!/bin/sh

# Job name:
#SBATCH --job-name=
#
#
#SBATCH --partition=batch
# Request nodes:
#SBATCH --nodes=1
#
# Specify number of tasks for use case (example):
#SBATCH --ntasks-per-node=1
#
# Processors per task:
#SBATCH --cpus-per-task=100
#
# Wall clock limit:
#SBATCH --time=2-12:00:00

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=BEGIN,END,FAIL

# send mail to this address
#SBATCH --mail-user=B.Rees@soton.ac.uk

# Send SIGUSR1 90 seconds before timeout
#SBATCH --signal=SIGUSR1@90 

#script

source /home/br7g18/miniconda3/bin/activate simopt

#Run - this is added in a script
