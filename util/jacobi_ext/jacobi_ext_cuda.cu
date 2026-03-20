    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    // ─────────────────────────────────────────────────────────────────────────────
    // Forward declarations
    // ─────────────────────────────────────────────────────────────────────────────
    template<int ITER>
    __global__ void jacobi_kernel_fused(
        const float* x_in,       // NOT __restrict__: we alias with x_out mid-kernel
            float* x_out,
        const float* __restrict__ bc_value,
        const float* __restrict__ bc_mask,
        const float* __restrict__ f,
        int N, int C, int H, int W
    );

    // ─────────────────────────────────────────────────────────────────────────────
    // Host wrapper
    // ─────────────────────────────────────────────────────────────────────────────
    torch::Tensor jacobi_forward_cuda(
        torch::Tensor x,
        torch::Tensor bc_value,
        torch::Tensor bc_mask,
        c10::optional<torch::Tensor> f,
        int num_iters
    ) {
        x        = x.contiguous();
        bc_value = bc_value.contiguous();
        bc_mask  = bc_mask.contiguous();

        torch::Tensor f_tensor;
        if (f.has_value())
            f_tensor = f.value().contiguous();

        // Two buffers: kernel ping-pongs between them across iterations.
        // We always return buf[final_buf] to the caller.
        torch::Tensor buf0 = x;                    // initial input
        torch::Tensor buf1 = torch::empty_like(x); // scratch

        int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);

        const float* f_ptr = f_tensor.defined() ? f_tensor.data_ptr<float>() : nullptr;
        const float* bc_v  = bc_value.data_ptr<float>();
        const float* bc_m  = bc_mask.data_ptr<float>();

        const int TILE = 16;
        dim3 threads(TILE, TILE);
        dim3 blocks((W + TILE-1)/TILE, (H + TILE-1)/TILE, N * C);

        // Each kernel call does ONE correct Jacobi iteration.
        // We launch num_iters times, alternating read/write buffers.
        // Benefits vs plain conv2d:
        //   - bc_mask / bc_value / f are pre-fetched into registers once per
        //     iteration (not 3 separate kernels)
        //   - no separate elementwise kernels for BC application or f subtraction
        //   - kernel launch overhead: num_iters launches vs 3*num_iters launches
        //
        // NOTE: true "wavefront" temporal blocking (keeping intermediate values
        // in shared memory across iterations) is mathematically correct only if
        // the valid compute region shrinks by 1 pixel per iteration, which makes
        // it much more complex.  That optimization is left for future work.
        // The current approach is correct and still gives meaningful speedup.

        for (int iter = 0; iter < num_iters; iter++) {
            const float* src = (iter % 2 == 0) ? buf0.data_ptr<float>()
                                            : buf1.data_ptr<float>();
                float* dst = (iter % 2 == 0) ? buf1.data_ptr<float>()
                                            : buf0.data_ptr<float>();

            jacobi_kernel_fused<1><<<blocks, threads>>>(
                src, dst, bc_v, bc_m, f_ptr, N, C, H, W);
        }

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
            "jacobi_kernel_fused launch failed: ", cudaGetErrorString(err));

        // Return whichever buffer was written last
        return (num_iters % 2 == 0) ? buf0 : buf1;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Kernel — single Jacobi step with fused BC and source term
    //
    // One kernel call = one Jacobi iteration.
    // Shared memory tile: threads load their pixel + halo, sync, then compute.
    // bc_mask, bc_value, f are read once per thread and kept in registers.
    // The template parameter ITER is fixed at 1 here; the loop is in the host
    // wrapper above.  The template is kept so future wavefront work can reuse
    // the same structure.
    // ─────────────────────────────────────────────────────────────────────────────
    template<int ITER>
    __global__ void jacobi_kernel_fused(
        const float* x_in,
            float* x_out,
        const float* __restrict__ bc_value,
        const float* __restrict__ bc_mask,
        const float* __restrict__ f,
        int N, int C, int H, int W
    ) {
        const int TILE = 16;
        __shared__ float tile[TILE+2][TILE+2];

        int tx = threadIdx.x, ty = threadIdx.y;
        int j  = blockIdx.x * TILE + tx;
        int i  = blockIdx.y * TILE + ty;
        int nc = blockIdx.z;
        int n  = nc / C,  c = nc % C;

        bool valid = (i < H && j < W);
        int base   = ((n * C + c) * H + i) * W + j;

        // ── Load values that don't change across iterations into registers ────────
        float mask_val = valid ? bc_mask[base]  : 0.f;
        float bc_val   = valid ? bc_value[base] : 0.f;
        float f_val    = (f && valid) ? f[base] : 0.f;

        // ── Load tile interior ────────────────────────────────────────────────────
        tile[ty+1][tx+1] = valid ? x_in[base] : 0.f;

        // ── Load halo ────────────────────────────────────────────────────────────
        if (tx == 0)
            tile[ty+1][0]      = (j > 0   && valid) ? x_in[base - 1] : 0.f;
        if (tx == TILE-1)
            tile[ty+1][TILE+1] = (j < W-1 && valid) ? x_in[base + 1] : 0.f;
        if (ty == 0)
            tile[0][tx+1]      = (i > 0   && valid) ? x_in[base - W] : 0.f;
        if (ty == TILE-1)
            tile[TILE+1][tx+1] = (i < H-1 && valid) ? x_in[base + W] : 0.f;

        __syncthreads();

        if (!valid) return;

        float val;

        if (mask_val > 0.5f) {
            // BC pixel: prescribed value, no stencil needed
            val = bc_val;
        } else if (i == 0 || j == 0 || i == H-1 || j == W-1) {
            // Grid boundary (no BC prescribed): copy input unchanged
            val = x_in[base];
        } else {
            // Interior: 5-point Jacobi stencil
            val = 0.25f * (
                tile[ty  ][tx+1] +   // up
                tile[ty+2][tx+1] +   // down
                tile[ty+1][tx  ] +   // left
                tile[ty+1][tx+2]     // right
            ) - 0.25f * f_val;
        }

        x_out[base] = val;
    }
