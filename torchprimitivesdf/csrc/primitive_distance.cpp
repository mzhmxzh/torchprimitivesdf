#include <ATen/ATen.h>
#include <cstdio>
#include "check.h"

using namespace at::indexing;

namespace primitive {

#ifdef WITH_CUDA

void box_distance_forward_cuda_impl(
    at::Tensor points, 
    at::Tensor box, 
    at::Tensor distances, 
    at::Tensor dis_signs, 
    at::Tensor closest_points);

void box_distance_backward_cuda_impl(
    at::Tensor grad_distances, 
    at::Tensor points, 
    at::Tensor closest_points, 
    at::Tensor grad_points);

void transform_points_inverse_forward_cuda_impl(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed);

void transform_points_inverse_backward_cuda_impl(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points,
    at::Tensor grad_translations,
    at::Tensor grad_rotations);

void fixed_transform_points_inverse_forward_cuda_impl(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed);

void fixed_transform_points_inverse_backward_cuda_impl(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points);

#endif  // WITH_CUDA

void box_distance_forward_cuda(
    at::Tensor points, 
    at::Tensor box, 
    at::Tensor distances, 
    at::Tensor dis_signs, 
    at::Tensor closest_points) {
    CHECK_CUDA(points);
    CHECK_CUDA(box);
    CHECK_CUDA(distances);
    CHECK_CUDA(dis_signs);
    CHECK_CUDA(closest_points);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(box);
    CHECK_CONTIGUOUS(distances);
    CHECK_CONTIGUOUS(dis_signs);
    CHECK_CONTIGUOUS(closest_points);
    const int num_points = points.size(0);
    CHECK_SIZES(points, num_points, 3);
    CHECK_SIZES(box, 3);
    CHECK_SIZES(distances, num_points);
    CHECK_SIZES(dis_signs, num_points);
    CHECK_SIZES(closest_points, num_points, 3);
#if WITH_CUDA
    box_distance_forward_cuda_impl(points, box, distances, dis_signs, closest_points);
#else
    AT_ERROR("box_distance not built with CUDA");
#endif
}

void box_distance_backward_cuda(
    at::Tensor grad_distances, 
    at::Tensor points, 
    at::Tensor closest_points, 
    at::Tensor grad_points) {
    CHECK_CUDA(grad_distances);
    CHECK_CUDA(points);
    CHECK_CUDA(closest_points);
    CHECK_CUDA(grad_points);
    CHECK_CONTIGUOUS(grad_distances);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(closest_points);
    CHECK_CONTIGUOUS(grad_points);
    const int num_points = points.size(0);
    CHECK_SIZES(grad_distances, num_points);
    CHECK_SIZES(points, num_points, 3);
    CHECK_SIZES(closest_points, num_points, 3);
    CHECK_SIZES(grad_points, num_points, 3);

#if WITH_CUDA
    box_distance_backward_cuda_impl(grad_distances, points, closest_points, grad_points);
#else
    AT_ERROR("box_distance_backward not built with CUDA");
#endif
}

void box_distance_forward(
    at::Tensor points, 
    at::Tensor box, 
    at::Tensor distances, 
    at::Tensor dis_signs, 
    at::Tensor closest_points) {
    const int num_points = points.size(0);
    CHECK_SIZES(points, num_points, 3);
    CHECK_SIZES(box, 3);
    CHECK_SIZES(distances, num_points);
    CHECK_SIZES(dis_signs, num_points);
    CHECK_SIZES(closest_points, num_points, 3);

    auto q = points.abs() - box;
    auto tmp = q.max(-1);
    distances.set_(at::clamp_min(q, 0).square().sum(-1) + at::clamp_max(std::get<0>(tmp), 0).square());
    dis_signs.set_((q > 0).any(-1));
    closest_points.set_(at::maximum(at::minimum(points, box), -box));
    // auto residual = at::zeros_like(points);
    // auto arange = at::arange(points.size(0), c10::kLong, c10::kStrided, points.device(), false);
    // residual.index_put_({arange, std::get<1>(tmp)}, at::clamp_max(std::get<0>(tmp), 0));
    // residual.set_(at::where((points.index({arange, std::get<1>(tmp)}) > 0).unsqueeze(1), -residual, residual));
    // closest_points.set_(closest_points + residual);
    auto residual = at::clamp_max(std::get<0>(tmp), 0);
    auto arange = at::arange(points.size(0), c10::kLong, c10::kStrided, points.device(), false);
    residual.set_(at::where((points.index({arange, std::get<1>(tmp)}) > 0), -residual, residual));
    closest_points.index_put_({arange, std::get<1>(tmp)}, closest_points.index({arange, std::get<1>(tmp)}) + residual);
    // auto face_indices = (std::get<1>(tmp)).index({~dis_signs});  // [n_interiors]
    // auto face_indices = at::zeros({(~dis_signs).sum().item().toInt()}, c10::kLong, c10::kStrided, points.device(), false);
    // auto face = box.index({face_indices});
    // closest_points.index_put_({~dis_signs, face_indices}, at::where(points.index({~dis_signs, face_indices}) > 0, face, -face));
}

void box_distance_backward(
    at::Tensor grad_distances, 
    at::Tensor points, 
    at::Tensor closest_points, 
    at::Tensor grad_points) {
    const int num_points = points.size(0);
    CHECK_SIZES(grad_distances, num_points);
    CHECK_SIZES(points, num_points, 3);
    CHECK_SIZES(closest_points, num_points, 3);
    CHECK_SIZES(grad_points, num_points, 3);

    grad_points.set_(2 * grad_distances.unsqueeze(1) * (points - closest_points));
}

void transform_points_inverse_forward_cuda(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed) {
    CHECK_CUDA(points);
    CHECK_CUDA(translations);
    CHECK_CUDA(rotations);
    CHECK_CUDA(points_transformed);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(translations);
    CHECK_CONTIGUOUS(rotations);
    CHECK_CONTIGUOUS(points_transformed);
    const int batch_size = points.size(0);
    const int num_points = points.size(1);
    CHECK_SIZES(points, batch_size, num_points, 3);
    CHECK_SIZES(translations, batch_size, 3);
    CHECK_SIZES(rotations, batch_size, 3, 3);
    CHECK_SIZES(points_transformed, batch_size, num_points, 3);

#if WITH_CUDA
    transform_points_inverse_forward_cuda_impl(points, translations, rotations, points_transformed);
#else
    AT_ERROR("transform_points_inverse_forward not built with CUDA");
#endif
}

void transform_points_inverse_backward_cuda(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points,
    at::Tensor grad_translations,
    at::Tensor grad_rotations) {
    
    CHECK_CUDA(grad_points_transformed);
    CHECK_CUDA(points);
    CHECK_CUDA(translations);
    CHECK_CUDA(rotations);
    CHECK_CUDA(grad_points);
    CHECK_CUDA(grad_translations);
    CHECK_CUDA(grad_rotations);
    CHECK_CONTIGUOUS(grad_points_transformed);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(translations);
    CHECK_CONTIGUOUS(rotations);
    CHECK_CONTIGUOUS(grad_points);
    CHECK_CONTIGUOUS(grad_translations);
    CHECK_CONTIGUOUS(grad_rotations);
    const int batch_size = points.size(0);
    const int num_points = points.size(1);
    CHECK_SIZES(grad_points_transformed, batch_size, num_points, 3);
    CHECK_SIZES(points, batch_size, num_points, 3);
    CHECK_SIZES(translations, batch_size, 3);
    CHECK_SIZES(rotations, batch_size, 3, 3);
    CHECK_SIZES(grad_points, batch_size, num_points, 3);
    CHECK_SIZES(grad_translations, batch_size, 3);
    CHECK_SIZES(grad_rotations, batch_size, 3, 3);

#if WITH_CUDA
    transform_points_inverse_backward_cuda_impl(grad_points_transformed, points, translations, rotations, grad_points, grad_translations, grad_rotations);
#else
    AT_ERROR("transform_points_inverse_backward not built with CUDA");
#endif
}

void transform_points_inverse_forward(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed) {
    points_transformed.set_(at::bmm(points - translations.unsqueeze(1), rotations));
}

void transform_points_inverse_backward(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points,
    at::Tensor grad_translations,
    at::Tensor grad_rotations) {
    grad_points.set_(at::bmm(grad_points_transformed, rotations.transpose(1, 2)));
    grad_rotations.set_(at::bmm((points - translations.unsqueeze(1)).transpose(1, 2), grad_points_transformed));
    grad_translations.set_(-grad_points.sum(1));
}

void fixed_transform_points_inverse_forward_cuda(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed) {
    CHECK_CUDA(points);
    CHECK_CUDA(translations);
    CHECK_CUDA(rotations);
    CHECK_CUDA(points_transformed);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(translations);
    CHECK_CONTIGUOUS(rotations);
    CHECK_CONTIGUOUS(points_transformed);
    const int num_points = points.size(0);
    CHECK_SIZES(points, num_points, 3);
    CHECK_SIZES(translations, 3);
    CHECK_SIZES(rotations, 3, 3);
    CHECK_SIZES(points_transformed, num_points, 3);

#if WITH_CUDA
    fixed_transform_points_inverse_forward_cuda_impl(points, translations, rotations, points_transformed);
#else
    AT_ERROR("fixed_transform_points_inverse_forward not built with CUDA");
#endif
}

void fixed_transform_points_inverse_backward_cuda(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points) {
    CHECK_CUDA(grad_points_transformed);
    CHECK_CUDA(points);
    CHECK_CUDA(translations);
    CHECK_CUDA(rotations);
    CHECK_CUDA(grad_points);
    CHECK_CONTIGUOUS(grad_points_transformed);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(translations);
    CHECK_CONTIGUOUS(rotations);
    CHECK_CONTIGUOUS(grad_points);
    const int num_points = points.size(0);
    CHECK_SIZES(grad_points_transformed, num_points, 3);
    CHECK_SIZES(points, num_points, 3);
    CHECK_SIZES(translations, 3);
    CHECK_SIZES(rotations, 3, 3);
    CHECK_SIZES(grad_points, num_points, 3);

#if WITH_CUDA
    fixed_transform_points_inverse_backward_cuda_impl(grad_points_transformed, points, translations, rotations, grad_points);
#else
    AT_ERROR("fixed_transform_points_inverse_backward not built with CUDA");
#endif
}

void fixed_transform_points_inverse_forward(
    at::Tensor points,
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor points_transformed) {
    points_transformed.set_(at::bmm(points - translations, rotations));
}

void fixed_transform_points_inverse_backward(
    at::Tensor grad_points_transformed, 
    at::Tensor points, 
    at::Tensor translations,
    at::Tensor rotations,
    at::Tensor grad_points) {
    grad_points.set_(at::bmm(grad_points_transformed, rotations.transpose(0, 1)));
}

}  // namespace primitive
