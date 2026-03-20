import torch
import torch.nn.functional as F
import numpy as np

try:
    from jacobi import jacobi_step
    HAS_CUSTOM = True
except ImportError:
    print("WARNING: Could not import jacobi_step. Custom implementation will be skipped.")
    HAS_CUSTOM = False


# ── Kernel is built ONCE per channel count, not inside the timed loop ──────────
_kernel_cache = {}

def get_kernel(channels, device):
    key = (channels, str(device))
    if key not in _kernel_cache:
        k = torch.tensor([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=torch.float32, device=device)
        _kernel_cache[key] = (k.view(1, 1, 3, 3) / 4.0).repeat(channels, 1, 1, 1)
    return _kernel_cache[key]


def jacobi_conv2d(x, bc_value, bc_mask, f, kernel, iters):
    """Reference: kernel is pre-built and passed in."""
    C = x.size(1)
    result = x
    for _ in range(iters):
        result = F.conv2d(result, kernel, padding=1, groups=C)
        if f is not None:
            result = result - 0.25 * f
        result = torch.where(bc_mask > 0.5, bc_value, result)
    return result


def cuda_event_benchmark(func, *args, warmup=20, iters=200):
    """
    Correct GPU timing using CUDA events.

    time.perf_counter() measures CPU wall time and returns immediately
    before the GPU has finished — CUDA kernels execute asynchronously.
    CUDA events are inserted directly into the GPU command stream,
    so elapsed_time() gives true GPU execution time.
    """
    # Warmup: ensures cuDNN autotuning, JIT compilation etc. are done
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        func(*args)
    end.record()

    torch.cuda.synchronize()  # wait for end event to be recorded

    return start.elapsed_time(end) / iters  # ms per call


def create_problem(batch, channels, height, width, device='cuda'):
    x        = torch.zeros(batch, channels, height, width, device=device)
    bc_mask  = torch.zeros_like(x)
    bc_value = torch.zeros_like(x)

    bc_mask[:, :,  0, :] = 1
    bc_mask[:, :, -1, :] = 1
    bc_mask[:, :, :,  0] = 1
    bc_mask[:, :, :, -1] = 1
    bc_value = bc_mask.clone()

    f = torch.randn_like(x) * 0.1
    return x, bc_value, bc_mask, f


def run_benchmark(configs, jacobi_iters=4):
    print(f"\n{'='*72}")
    print(f"BENCHMARK  —  {jacobi_iters} Jacobi iterations, CUDA event timing")
    print(f"{'='*72}")
    print(f"{'Config':<22} {'Conv2d (ms)':>12} {'Custom (ms)':>12} {'Speedup':>10}")
    print(f"{'─'*72}")

    rows = []
    for (batch, channels, height, width) in configs:
        x, bc_value, bc_mask, f = create_problem(batch, channels, height, width)
        kernel = get_kernel(channels, x.device)

        t_conv = cuda_event_benchmark(
            jacobi_conv2d, x, bc_value, bc_mask, f, kernel, jacobi_iters)

        label = f"{batch}x{channels}x{height}x{width}"

        if HAS_CUSTOM:
            t_cust = cuda_event_benchmark(
                jacobi_step, x, bc_value, bc_mask, f, jacobi_iters)
            speedup = t_conv / t_cust
            print(f"{label:<22} {t_conv:>12.4f} {t_cust:>12.4f} {speedup:>9.2f}x")
            rows.append(speedup)
        else:
            print(f"{label:<22} {t_conv:>12.4f} {'N/A':>12} {'N/A':>10}")

    if rows:
        print(f"{'─'*72}")
        print(f"{'Geometric mean speedup':<46} {np.exp(np.mean(np.log(rows))):>9.2f}x")

    print()


def test_correctness(jacobi_iters=4):
    print(f"\n{'='*72}")
    print("CORRECTNESS CHECK  —  custom vs conv2d")
    print(f"{'='*72}")

    batch, channels, height, width = 2, 3, 64, 64
    x, bc_value, bc_mask, f = create_problem(batch, channels, height, width)
    kernel = get_kernel(channels, x.device)

    ref = jacobi_conv2d(x, bc_value, bc_mask, f, kernel, jacobi_iters)

    if not HAS_CUSTOM:
        print("Custom implementation not available.")
        return

    out = jacobi_step(x, bc_value, bc_mask, f, jacobi_iters)
    diff = (ref - out).abs()
    print(f"  Max  |ref - custom|: {diff.max().item():.2e}")
    print(f"  Mean |ref - custom|: {diff.mean().item():.2e}")
    status = "PASS" if diff.max().item() < 1e-4 else "FAIL"
    print(f"  {status}")


def main():
    print("="*72)
    print(f"JACOBI BENCHMARK   GPU: {torch.cuda.get_device_name(0)}")
    print("="*72)

    test_correctness(jacobi_iters=4)

    configs = [
        (1, 1,   64,   64),
        (1, 1,  128,  128),
        (1, 1,  256,  256),
        (1, 1,  257,  257),
        (1, 1,  512,  512),
        (1, 1, 1024, 1024),
        (1, 1, 1025, 1025),
        (4, 3,  128,  128),
        (8, 3,  256,  256),
    ]

    # Speedup should grow with iters: more iterations = more DRAM traffic saved by fusion
    for iters in [1, 2, 4, 8]:
        run_benchmark(configs, jacobi_iters=iters)


if __name__ == "__main__":
    main()