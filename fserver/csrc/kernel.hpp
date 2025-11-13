#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>


torch::Tensor map_pinned_tensor(torch::Tensor tensor, int64_t device_index);
void write_flag(torch::Tensor flag, torch::Tensor seq);
void wait_flag(torch::Tensor flag, torch::Tensor seq);
void seq_add_one(torch::Tensor seq);

void pybind_kernel(py::module &m){
  // StepMesh utils
  m.def("map_pinned_tensor", &map_pinned_tensor, py::arg("tensor"), py::arg("device_index"));
  m.def("write_flag", &write_flag, py::arg("flag"), py::arg("seq"));
  m.def("wait_flag", &wait_flag, py::arg("flag"), py::arg("seq"));
  m.def("seq_add_one", &seq_add_one, py::arg("seq"));
}