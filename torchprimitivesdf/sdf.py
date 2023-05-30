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
