import torch
import jacobi_ext

def jacobi_step(x, bc_value, bc_mask, f=None, iters=1):
    return jacobi_ext.forward(x, bc_value, bc_mask, f, iters)
