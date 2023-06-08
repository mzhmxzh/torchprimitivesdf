#include <torch/extension.h>
#include "primitive_distance.h"

namespace primitive {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("box_distance_forward_cuda", &box_distance_forward_cuda);
  m.def("box_distance_backward_cuda", &box_distance_backward_cuda);
  m.def("box_distance_forward", &box_distance_forward);
  m.def("box_distance_backward", &box_distance_backward);
  m.def("transform_points_inverse_forward_cuda", &transform_points_inverse_forward_cuda);
  m.def("transform_points_inverse_backward_cuda", &transform_points_inverse_backward_cuda);
  m.def("transform_points_inverse_forward", &transform_points_inverse_forward);
  m.def("transform_points_inverse_backward", &transform_points_inverse_backward);
  m.def("fixed_transform_points_inverse_forward_cuda", &fixed_transform_points_inverse_forward_cuda);
  m.def("fixed_transform_points_inverse_backward_cuda", &fixed_transform_points_inverse_backward_cuda);
  m.def("fixed_transform_points_inverse_forward", &fixed_transform_points_inverse_forward);
  m.def("fixed_transform_points_inverse_backward", &fixed_transform_points_inverse_backward);
}

}  // namespace primitive
