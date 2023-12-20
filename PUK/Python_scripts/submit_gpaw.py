#!/bin/sh 
#SBATCH --job-name=optimize_EtOH
#SBATCH --partition=kemi_gemma3 
#SBATCH --output=seq.%j.out 
#SBATCH --error=seq.%j.err 
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-core=2
#SBATCH --mem=8G

export OMP_NUM_THREADS=2
export GPAW_SETUP_PATH=/kemi/williamb/opt/gpaw-datasets/gpaw-setups-0.9.20000/

mpirun gpaw -P 8 python ./opt_EtOH.py