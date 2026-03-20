#!/bin/bash


# biharmonic stencil
python.exe train.py  --structure "unet"  --downsampling_policy "lerp"  --upsampling_policy "lerp"  --num_iterations 16  --relative_tolerance 1e-6  --initialize_x0 "random"  --num_mg_layers 4  --num_mg_pre_smoothing 3  --num_mg_post_smoothing 3  --biharmonic_solver "biharmonic_stencil" --activation "none"  --initialize_trainable_parameters "default"  --optimizer "adam"  --scheduler "step" "50" "0.1"  --initial_lr 1e-3  --lambda_1 1  --lambda_2 1  --start_epoch 0  --max_epoch 100  --save_every 10  --evaluate_every 1  --checkpoint_root "./checkpoint"  --dataset_root "../SynDat_1025"  --num_workers 8  --batch_size 2  --use_data 0.025 --seed 9590589012167207234

# stacked poisson
python.exe train.py  --structure "unet"  --downsampling_policy "lerp"  --upsampling_policy "lerp"  --num_iterations 16  --relative_tolerance 1e-6  --initialize_x0 "random"  --num_mg_layers 4  --num_mg_pre_smoothing 3  --num_mg_post_smoothing 3  --biharmonic_solver "stacked_poisson" --activation "none"  --initialize_trainable_parameters "default"  --optimizer "adam"  --scheduler "step" "50" "0.1"  --initial_lr 1e-3  --lambda_1 1  --lambda_2 1  --start_epoch 0  --max_epoch 100  --save_every 10  --evaluate_every 1  --checkpoint_root "./checkpoint"  --dataset_root "../SynDat_1025"  --num_workers 8  --batch_size 2  --use_data 0.025 --seed 9590589012167207234
