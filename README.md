### Description
A  thesis based on this reposit repository.  
Eastablish the ransac process on the light field networks to reconstruct occluded images

change:config.py for saving files
gpus:1
sparsity to choose initial random image pixels

In the experiment, the test is tested on each instance separately which means one file name in softras_test.lst. And the batch_size is set to 1 for one instance.
One instace can have multiple iamges with different poses.
Note that the rec_nmr.py script uses the viewlist under ./experiment_scripts/viewlists/src_dvr.txt to pick which views to reconstruct the objects from.


### Command
typical command on gpus on the fau woody cluster: 
srun --mpi=pmi2 python3 experiment_scripts/ransac.py --data_root=path_to_nmr_dataset --dataset=NMR  --checkpoint=path_to_training_checkpoint --experiment_name=reconstruct_name --test_experiment_name=test_name --gpus=1 --sparsity= the number you want to sparsify(typical 64)
