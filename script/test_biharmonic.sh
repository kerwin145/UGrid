#!/bin/bash


python test.py --checkpoint_root "./checkpoint" --load_experiment "20251207-111229" --load_epoch 100 --dataset_root "../../Downloads/SynDat1025" --num_workers 8 --batch_size 1 --seed 9590589012167207234 --biharmonic_problem
