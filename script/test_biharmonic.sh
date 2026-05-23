#!/bin/bash


python test.py --checkpoint_root "./checkpoint" --load_experiment "biharmonic_9" --load_epoch 150 --dataset_root "../../Downloads/SynDat1025" --num_workers 8 --batch_size 1 --seed 9590589012167207234 --biharmonic_solver "biharmonic_stencil" 

python test.py --checkpoint_root "./checkpoint" --load_experiment "stacked_poisson2" --load_epoch 100 --dataset_root "../../Downloads/SynDat1025" --num_workers 8 --batch_size 1 --seed 9590589012167207234 --biharmonic_solver "stacked_poisson"
