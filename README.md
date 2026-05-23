# UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs

This is a repository forked from [Ugrid](https://github.com/AXIHIXA/UGrid)

## Biharmonic extension
- ```jacobi_residuals.ipynb```, we see how the residual norm progresses through jacobi iterations for 3x3 and 5x5 laplace stencil, as well as the 5x5 biharmonic stencil
- In checkpoint folder, there are 9 available checkpoints, where I tested different hyper parameter settings. ```biharmonic_3``` performed the best

## Hardware aware iterators (testing)
Compiling the hardware aware iterators
- Ensure cuda 11.8 is installed
- Open x64 Native Tools Command Prompt
  - Make sure MSVC toolset (v143, 14.3x) is installed, so as to be compatible with 11.8
- Activate conda environment
- Run this inside the same command prompt: ```set DISTUTILS_USE_SDK=1```
- ```python -m pip install -v -e .```
- This was more so testing out fused kernels. Not much else here

## Stacked poisson (testing)
- We tried stacking doing two stacked poisson solvers for biharmonic, but due to errors not linearly propogating, this did not perform to well.
- More theory and work could be needed here to have the stacked solver work.

### arg
Added param for biharmonic_problem for test and train

### checkpoint
Contains the trained model. The one I used is ```20251207-111229```
Notes on hyper parameters are inlcuded in the folder

### ./model/ugrid.py
Configured functions to take in a biharmonic true/false parameter. Adjusted the forward function calls for Ugrid and Unet classes to fit biharmonic equation solving.

### ./script
Added training and test scripts for biharmonic, and also colored test scripts for poisson and biharmonic. You can run the testcases by running these scripts!

### ./util
kernel.py: added biharmonic stencils and biharomnic jacobi iteration function. Also adjusted residual functions to accomodate biharmonic equations.
prepartion.py: added method to assemble biharmonic matrix to test the pde 

### ./var
Added a folder for colored test cases (drawn with ms paint)
Also contains the trained model for poisson at ```./var/checkpoint/22```

### ./visualized_output
Contains outputs for the runs for poisson, biharmonic, and the RGB versions

### test.py
Added ```test_on_single_data_color``` function for poisson and biharmonic questions

### Playground.ipynb 
Includes testing for the biharmonic stencil, and reading in images for the colored test cases and guassian blurring

Additional sources used:
Han, X. (2025). Physics-informed hardware-aware neural numerical solvers for differential equations: Theory, architecture, algorithms, and application (Doctoral dissertation). Stony Brook University.

---

This repository is the official implementation of our ICML'2024 paper

Xi Han, Fei Hou, Hong Qin, 
"UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs",
In *Proceedings of the 41st International Conference on Machine Learning*, 
pp. 17354--17373, July 2024. 

The paper is available at:
- [arXiv](https://arxiv.org/abs/2408.04846)
- [PMLR](https://proceedings.mlr.press/v235/han24a.html)
- [openreview](https://openreview.net/forum?id=vFATIZXlCm)

## Data Generation

To generate the dataset, run this command:

```bash
bash ./script/generate.sh
```

Please modify `generate.sh` to generate the `train`, `evaluate` and `test` datasets of the desired size. 

## Training

To train the model(s) in the paper, run this command:

```bash
bash ./script/train.sh
```

## Evaluation and Testing

To re-produce the testing results of UGrid, please run this command: 

```bash
bash ./script/test.sh
```

To compare with 
[AMGCL](https://github.com/ddemidov/amgcl) and [NVIDIA AmgX](https://developer.nvidia.com/amgx), 
please first compile the Python bindings for AMGCL and AmgX (see `./comparasion/cpmg/`), 
then run the following command:

```bash
bash ./script/compare.sh
```

To compare with [(Hsieh et al., 2019)](https://openreview.net/forum?id=rklaWn0qK7), 
please refer to [their offical repository](https://github.com/ermongroup/Neural-PDE-Solver). 

## Pre-trained Models

Self-contained in `var/checkpoint/22/`. 

## How to Cite

```
@inproceedings{han24-icml-ugrid,
  author={Han, Xi and Hou, Fei and Qin, Hong},
  title={{UGrid: An Efficient-And-Rigorous Neural Multigrid Solver for Linear PDEs}},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  volume={235},
  number={},
  pages={17354--17373},
  month={July},
  year={2024},
  url={https://arxiv.org/abs/2408.04846}
}
```
