#!/bin/bash -l
#
# allocate 1 node (4 Cores) for 6 hours
#PBS -l nodes=1:ppn=8:gtx1080,walltime=06:00:00
#
# job name 
#PBS -N Sparsejob_33
#
# first non-empty non-comment line ends PBS options

#conda activate
conda activate lf
#load required modules (compiler, ...)

#module load cuda/11.
# jobs always start in $HOME - 
# change to work directory
cd  ${PBS_O_WORKDIR}
#cd ./lfn/light-field-networks/
export OMP_NUM_THREADS=4
 
# run 
python ./experiment_scripts/train_single_class.py --data_root=./data/cars_train.hdf5 --experiment_name=car

