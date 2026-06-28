#!/bin/bash
#SBATCH --job-name=relaxation_serial
#SBATCH --output=out/out_%j.out
#SBATCH --error=err%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=short
#SBATCH --time=02:00:00

# 1. Load Modules
# module purge
# # # module load ohpc
# # module load gcc/13.3.0

# module load openmpi/5.0.3
# module load oneapi/2024.0/tbb
# module load oneapi/2024.0/compiler-rt
# module load oneapi/2024.0/mkl/latest

# Tell OpenMP to use the cores reserved by Slurm
# For a serial job, restrict threads to 1
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1

# 1. Set MKL threads to match the requested SLURM CPUs
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export JULIA_PROJECT=@.

# Run serial Julia under Slurm allocation
# Use srun so Slurm tracks the job; this runs the script in the current working directory
# srun julia --sysimage EulerAP.so --project=. equations/relaxation_euler2d.jl

# srun julia --project=. examples/relaxation_euler_1d/relaxation_euler_1d_sinosidal_riemann.jl
# srun julia --project=. examples/relaxation_euler_1d/relaxation_euler_1d_riemann.jl
# srun julia --project=. examples/relaxation_euler_1d/relaxation_euler_1d_barenblatt_convergence.jl
srun julia --project=. examples/relaxation_euler_1d/relaxation_euler_1d_barenblatt.jl