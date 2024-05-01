#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -A strachan
#SBATCH -N 1
#SBATCH --tasks-per-node=64
#SBATCH --job-name=diffusion-couple

set echo

#module load gcc/9.3.0
#module load openmpi/3.1.4

#module load lammps/20201029

module load anaconda/2022.10-py39
source activate ovito3.10

cd $SLURM_SUBMIT_DIR

python run.py

conda deactivate