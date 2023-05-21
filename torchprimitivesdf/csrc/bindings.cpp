#include <torch/extension.h>
#include "primitive_distance.h"

namespace primitive {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("box_distance_forward_cuda", &box_distance_forward_cuda);
  m.def("box_distance_backward_cuda", &box_distance_backward_cuda);
  m.def("box_distance_forward", &box_distance_forward);
  m.def("box_distance_backward", &box_distance_backward);
}

}  // namespace primitive
