#!/bin/sh 
#SBATCH --job-name=optimize_EtOH
#SBATCH --partition=kemi_gemma3 
#SBATCH --output=seq.%j.out 
#SBATCH --error=seq.%j.err 
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-core=2
#SBATCH --mem=8G

export OMP_NUM_THREADS=1
export OMPI_MCA_pml=^ucx
export OMP_MCA_osc=^ucx
mpirun --mca btl_openib_rroce_enable 1 gpaw -P 10 python ./opt_EtOH.py