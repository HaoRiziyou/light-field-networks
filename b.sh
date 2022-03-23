#!/bin/bash -l
#
# start 2 MPI processes
# allocate nodes for 24 hours
#SBATCH --time=24:00:00
# allocated 4 GPU (type not specified)
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=a100
# number of nodes and task
#SBATCH --nodes=1             
# job name 
#SBATCH --job-name=Rec_fullnmr_2gpu
# do not export environment variables
#SBATCH --export=NONE

#coonda
conda activate lf
# do not export environment variables
unset SLURM_EXPORT_ENV

srun --mpi=pmi2 python3 ./experiment_scripts/rec_nmr.py --data_root=/home/woody/iwi9/iwi9008h/data/NMR_Dataset/ --experiment_name=nmr_full_rec --gpus=2 --checkpoint_path=./data/pretrained/NMR_train.pth

