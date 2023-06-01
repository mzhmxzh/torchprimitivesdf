import torch
from torchprimitivesdf import _C

def box_sdf(points, box):
    """
    Calculate signed distances from points to box in box frame
    
    Interiors are negative, exteriors are positive
    
    Parameters
    ----------
    points: (N, 3) torch.Tensor
        points
    box: (3,) torch.Tensor
        box scales, [-box[0], box[0]] * [-box[1], box[1]] * [-box[2], box[2]]
    
    Returns
    -------
    distances: (N,) torch.Tensor
        squared distances from points to box
    dis_signs: (N,) torch.BoolTensor
        distance signs, externals are positive
    closest_points: (N, 3) torch.Tensor
        closest points on box surface
    """
    return _BoxDistanceCuda.apply(points, box)


class _BoxDistanceCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, box):
        num_points = points.shape[0]
        distances = torch.zeros([num_points], device=points.device, dtype=points.dtype)
        dis_signs = torch.zeros([num_points], device=points.device, dtype=torch.bool)
        closest_points = torch.zeros([num_points, 3], device=points.device, dtype=points.dtype)
        if points.is_cuda:
            _C.box_distance_forward_cuda(points, box, distances, dis_signs, closest_points)
        else:
            _C.box_distance_forward(points, box, distances, dis_signs, closest_points)
        # _C.box_distance_forward_cuda(points, box, distances, dis_signs, closest_points)
        # _C.box_distance_forward(points, box, distances, dis_signs, closest_points)
        ctx.save_for_backward(points.contiguous(), closest_points)
        ctx.mark_non_differentiable(dis_signs, closest_points)
        return distances, dis_signs, closest_points
    
    @staticmethod
    def backward(ctx, grad_distances, grad_dis_signs, grad_closest_points):
        points, closest_points = ctx.saved_tensors
        grad_distances = grad_distances.contiguous()
        grad_points = torch.zeros_like(points)
        grad_box = None
        # if points.is_cuda:
        #     _C.box_distance_backward_cuda(grad_distances, points, closest_points, grad_points)
        # else:
        #     _C.box_distance_backward(grad_distances, points, closest_points, grad_points)
        # _C.box_distance_backward_cuda(grad_distances, points, closest_points, grad_points)
        _C.box_distance_backward(grad_distances, points, closest_points, grad_points)
        return grad_points, grad_box


def transform_points_inverse(points, matrices):
    return _TransformPointsInverse.apply(points, matrices)


class _TransformPointsInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrices):
        if points.is_cuda:
            B, N, _ = points.shape
            points_reshaped = points.reshape(-1, 3)  # (B * N, 3)
            matrices_reshaped = matrices.repeat_interleave(N, dim=0)  # (B * N, 4, 4)
            points_transformed_reshaped = torch.zeros_like(points_reshaped)
            _C.transform_points_inverse_forward_cuda(points_reshaped, matrices_reshaped, points_transformed_reshaped)
            points_transformed = points_transformed_reshaped.reshape(B, N, 3)
        else:
            points_transformed = torch.zeros_like(points)
            _C.transform_points_inverse_forward(points, matrices, points_transformed)
        ctx.save_for_backward(points, matrices)
        return points_transformed

    @staticmethod
    def backward(ctx, grad_points_transformed):
        points, matrices = ctx.saved_tensors
        if points.is_cuda:
            B, N, _ = points.shape
            grad_points_transformed_reshaped = grad_points_transformed.reshape(-1, 3).contiguous()
            points_reshaped = points.reshape(-1, 3)
            matrices_reshaped = matrices.repeat_interleave(N, 0)
            grad_points_reshaped = torch.zeros_like(points_reshaped)
            grad_matrices_reshaped = torch.zeros_like(matrices_reshaped)
            _C.transform_points_inverse_backward_cuda(grad_points_transformed_reshaped, points_reshaped, matrices_reshaped, grad_points_reshaped, grad_matrices_reshaped)
            grad_points = grad_points_reshaped.reshape(B, N, 3)
            grad_matrices = grad_matrices_reshaped.reshape(B, N, 4, 4).sum(dim=1)
        else:
            grad_points = torch.zeros_like(points)
            grad_matrices = torch.zeros_like(matrices)
            _C.transform_points_inverse_backward(grad_points_transformed, points, matrices, grad_points, grad_matrices)
        return grad_points, grad_matrices
