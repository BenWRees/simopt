#!/bin/sh

# Job name:
#SBATCH --job-name=
#
# Partition:
#SBATCH --partition=batch
#
# Request nodes:
#SBATCH --nodes= 9
#
# Specify number of tasks for use case (example):
#SBATCH --ntasks-per-node=1
#
# Processors per task:
#SBATCH --cpus-per-task=50
#
# Wall clock limit:
#SBATCH --time=2-12:00:00

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=BEGIN,END,FAIL

# send mail to this address
#SBATCH --mail-user=B.Rees@soton.ac.uk

#script
source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh

#Load in Conda Environment
conda init
conda activate simopt

#Run - this is added in a script
