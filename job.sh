#!/bin/bash
#SBATCH --job-name=relaxation_serial
#SBATCH --output=out_relaxation/out_%j.out
#SBATCH --error=err_relaxation_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=short
#SBATCH --time=02:00:00

# 1. Load Modules
# module purge
# module load ohpc
# module load gcc/13.3.0

# module load openmpi/5.0.3
# module load oneapi/2024.0/tbb
# module load oneapi/2024.0/compiler-rt
# module load oneapi/2024.0/mkl/latest

# Tell OpenMP to use the cores reserved by Slurm
# For a serial job, restrict threads to 1
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
export JULIA_PROJECT=@.

# Run serial Julia under Slurm allocation
# Use srun so Slurm tracks the job; this runs the script in the current working directory
# srun julia --sysimage EulerAP.so --project=. equations/relaxation_euler2d.jl
srun julia --project=. equations/relaxation_euler2d.jl

