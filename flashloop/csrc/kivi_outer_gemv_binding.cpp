#include <torch/extension.h>

#include "kivi_outer_gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "gemv_forward_cuda_outer_dim_logical",
      &gemv_forward_cuda_outer_dim_logical,
      "KIVI-style outer-dimension 4-bit GEMV with a logical output width");
  module.def(
      "qk_selected_cuda",
      &qk_selected_cuda,
      "Selected-token QK over KIVI per-channel K storage");
  module.def(
      "pv_selected_cuda",
      &pv_selected_cuda,
      "Selected-token PV over KIVI per-token V storage");
  module.def(
      "qk_cross_loop_cuda",
      &qk_cross_loop_cuda,
      "Fused cross-loop selected QK over four KIVI streams");
  module.def(
      "pv_cross_loop_cuda",
      &pv_cross_loop_cuda,
      "Fused cross-loop selected PV over four KIVI streams");
  module.def(
      "sparse_attention_delta_cuda",
      &sparse_attention_delta_cuda,
      "Fused KIVI subset attention and cached-mass correction");
  module.def(
      "dense_source_attention_cuda",
      &dense_source_attention_cuda,
      "Fused KIVI dense source attention for Loop 1/2");
}
