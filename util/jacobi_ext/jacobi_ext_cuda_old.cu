#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// --------------------------------------------------------
// Kernel declaration (forward declaration before use)
// --------------------------------------------------------
template<int ITER>
__global__ void jacobi_kernel_fused(
    const float* x,
    const float* bc_value,
    const float* bc_mask,
    const float* f,
    float* out,
    int N, int C, int H, int W
);

// --------------------------------------------------------
// Fused Jacobi wrapper
// --------------------------------------------------------
torch::Tensor jacobi_forward_cuda(
    torch::Tensor x,
    torch::Tensor bc_value,
    torch::Tensor bc_mask,
    c10::optional<torch::Tensor> f,
    int num_iters
) {
    x = x.contiguous();
    bc_value = bc_value.contiguous();
    bc_mask = bc_mask.contiguous();

    torch::Tensor f_tensor;
    if (f.has_value())
        f_tensor = f.value().contiguous();
    else
        f_tensor = torch::Tensor();

    auto out = torch::empty_like(x);

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    const int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks(
        (W + TILE - 1) / TILE,
        (H + TILE - 1) / TILE,
        N * C
    );

    // Dispatch based on number of iterations
    // Using template to allow loop unrolling
    switch(num_iters) {
        case 1:
            jacobi_kernel_fused<1><<<blocks, threads>>>(
                x.data_ptr<float>(),
                bc_value.data_ptr<float>(),
                bc_mask.data_ptr<float>(),
                f_tensor.defined() ? f_tensor.data_ptr<float>() : nullptr,
                out.data_ptr<float>(),
                N, C, H, W
            );
            break;
        case 2:
            jacobi_kernel_fused<2><<<blocks, threads>>>(
                x.data_ptr<float>(),
                bc_value.data_ptr<float>(),
                bc_mask.data_ptr<float>(),
                f_tensor.defined() ? f_tensor.data_ptr<float>() : nullptr,
                out.data_ptr<float>(),
                N, C, H, W
            );
            break;
        case 3:
            jacobi_kernel_fused<3><<<blocks, threads>>>(
                x.data_ptr<float>(),
                bc_value.data_ptr<float>(),
                bc_mask.data_ptr<float>(),
                f_tensor.defined() ? f_tensor.data_ptr<float>() : nullptr,
                out.data_ptr<float>(),
                N, C, H, W
            );
            break;
        case 4:
            jacobi_kernel_fused<4><<<blocks, threads>>>(
                x.data_ptr<float>(),
                bc_value.data_ptr<float>(),
                bc_mask.data_ptr<float>(),
                f_tensor.defined() ? f_tensor.data_ptr<float>() : nullptr,
                out.data_ptr<float>(),
                N, C, H, W
            );
            break;
        case 5:
            jacobi_kernel_fused<5><<<blocks, threads>>>(
                x.data_ptr<float>(),
                bc_value.data_ptr<float>(),
                bc_mask.data_ptr<float>(),
                f_tensor.defined() ? f_tensor.data_ptr<float>() : nullptr,
                out.data_ptr<float>(),
                N, C, H, W
            );
            break;
        default:
            // For other iteration counts, default to 4
            jacobi_kernel_fused<4><<<blocks, threads>>>(
                x.data_ptr<float>(),
                bc_value.data_ptr<float>(),
                bc_mask.data_ptr<float>(),
                f_tensor.defined() ? f_tensor.data_ptr<float>() : nullptr,
                out.data_ptr<float>(),
                N, C, H, W
            );
            break;
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    return out;
}

// --------------------------------------------------------
// Kernel implementation
// --------------------------------------------------------
template<int ITER>
__global__ void jacobi_kernel_fused(
    const float* x,
    const float* bc_value,
    const float* bc_mask,
    const float* f,
    float* out,
    int N, int C, int H, int W
) {
    const int TILE = 16;
    __shared__ float tile[TILE+2][TILE+2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = blockIdx.x * TILE + tx;
    int i = blockIdx.y * TILE + ty;
    int nc = blockIdx.z;  // combined N*C index
    
    int n = nc / C;
    int c = nc % C;

    // Calculate base index for this pixel in NCHW layout
    int base = ((n * C + c) * H + i) * W + j;

    // Bounds check
    if (i >= H || j >= W) return;

    // Load tile with halo
    tile[ty+1][tx+1] = x[base];

    // Load halo regions (boundary pixels)
    if (tx == 0 && j > 0)
        tile[ty+1][0] = x[base - 1];
    if (tx == TILE-1 && j < W-1)
        tile[ty+1][TILE+1] = x[base + 1];
    if (ty == 0 && i > 0)
        tile[0][tx+1] = x[base - W];
    if (ty == TILE-1 && i < H-1)
        tile[TILE+1][tx+1] = x[base + W];

    __syncthreads();

    // Interior points only compute Jacobi iteration
    if (i > 0 && j > 0 && i < H-1 && j < W-1) {
        float val = tile[ty+1][tx+1];

        #pragma unroll
        for (int k = 0; k < ITER; k++) {
            // 5-point stencil: (up + down + left + right) / 4
            val = 0.25f * (
                tile[ty][tx+1] +      // up
                tile[ty+2][tx+1] +    // down
                tile[ty+1][tx] +      // left
                tile[ty+1][tx+2]      // right
            );

            // Source term for Poisson equation: Laplacian(u) = f
            // So u = (neighbors_sum - h^2 * f) / 4
            // For h=1, this becomes: u = neighbors_sum/4 - f/4
            if (f)
                val -= 0.25f * f[base];

            // Update shared memory for next iteration
            __syncthreads();
            tile[ty+1][tx+1] = val;
            __syncthreads();
        }

        // Apply boundary conditions if this pixel is a BC pixel
        if (bc_mask[base] > 0.5f) {
            val = bc_value[base];
        }

        out[base] = val;
    } else {
        // Boundary pixels: just copy input or apply BC
        if (bc_mask[base] > 0.5f) {
            out[base] = bc_value[base];
        } else {
            out[base] = x[base];
        }
    }
}
