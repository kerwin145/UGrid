#include <torch/extension.h>
#include <vector>

torch::Tensor jacobi_forward_cuda(
    torch::Tensor x,
    torch::Tensor bc_value,
    torch::Tensor bc_mask,
    c10::optional<torch::Tensor> f,
    int iters
);

torch::Tensor jacobi_forward(
    torch::Tensor x,
    torch::Tensor bc_value,
    torch::Tensor bc_mask,
    c10::optional<torch::Tensor> f,
    int iters
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");

    return jacobi_forward_cuda(x, bc_value, bc_mask, f, iters);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &jacobi_forward,
          "Jacobi forward (CUDA)",
          py::arg("x"),
          py::arg("bc_value"),
          py::arg("bc_mask"),
          py::arg("f") = c10::nullopt,
          py::arg("iters") = 1);
}

