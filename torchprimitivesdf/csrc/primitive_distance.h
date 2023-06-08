#include <ATen/ATen.h>

namespace primitive {

void box_distance_forward_cuda(
    at::Tensor points, 
    at::Tensor box, 
    at::Tensor distances, 
    at::Tensor dis_signs, 
    at::Tensor closest_points);

void box_distance_backward_cuda(
    at::Tensor grad_distances, 
    at::Tensor points, 
    at::Tensor closest_points, 
    at::Tensor grad_points);

void box_distance_forward(
    at::Tensor points, 
    at::Tensor box, 
    at::Tensor distances, 
    at::Tensor dis_signs, 
    at::Tensor closest_points);

void box_distance_backward(
    at::Tensor grad_distances, 
    at::Tensor points, 
    at::Tensor closest_points, 
    at::Tensor grad_points);

void transform_points_inverse_forward_cuda(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed);

void transform_points_inverse_backward_cuda(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points,
    at::Tensor grad_translations,
    at::Tensor grad_rotations);

void transform_points_inverse_forward(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed);

void transform_points_inverse_backward(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points,
    at::Tensor grad_translations,
    at::Tensor grad_rotations);

void fixed_transform_points_inverse_forward_cuda(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed);

void fixed_transform_points_inverse_backward_cuda(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points);

void fixed_transform_points_inverse_forward(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed);

void fixed_transform_points_inverse_backward(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points);

}  // namespace primitive
