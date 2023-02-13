#!/bin/bash

#SBATCH --job-name="test job"
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu


export PYTHONUSERBASE=/scratch/hpc-prf-intexml/intexml6/.local
export PATH=/scratch/hpc-prf-intexml/intexml6/.local/bin:$PATH

module load lang
module load Python/3.10.4-GCCcore-11.3.0


python main.py
