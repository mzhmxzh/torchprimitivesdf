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
        if not points.is_contiguous():
            points = points.contiguous()
        if not box.is_contiguous():
            box = box.contiguous()
        num_points = points.shape[0]
        distances = torch.empty([num_points], device=points.device, dtype=points.dtype)
        dis_signs = torch.empty([num_points], device=points.device, dtype=torch.bool)
        closest_points = torch.empty([num_points, 3], device=points.device, dtype=points.dtype)
        if points.is_cuda:
            _C.box_distance_forward_cuda(points, box, distances, dis_signs, closest_points)
        else:
            _C.box_distance_forward(points, box, distances, dis_signs, closest_points)
        # _C.box_distance_forward_cuda(points, box, distances, dis_signs, closest_points)
        # _C.box_distance_forward(points, box, distances, dis_signs, closest_points)
        ctx.save_for_backward(points, closest_points)
        ctx.mark_non_differentiable(dis_signs, closest_points)
        return distances, dis_signs, closest_points
    
    @staticmethod
    def backward(ctx, grad_distances, grad_dis_signs, grad_closest_points):
        points, closest_points = ctx.saved_tensors
        if not grad_distances.is_contiguous():
            grad_distances = grad_distances.contiguous()
        grad_points = torch.empty_like(points)
        grad_box = None
        # if points.is_cuda:
        #     _C.box_distance_backward_cuda(grad_distances, points, closest_points, grad_points)
        # else:
        #     _C.box_distance_backward(grad_distances, points, closest_points, grad_points)
        # _C.box_distance_backward_cuda(grad_distances, points, closest_points, grad_points)
        _C.box_distance_backward(grad_distances, points, closest_points, grad_points)
        return grad_points, grad_box


def transform_points_inverse(points, translations, rotations):
    return _TransformPointsInverse.apply(points, translations, rotations)


class _TransformPointsInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, translations, rotations):
        if points.is_cuda:
            if not points.is_contiguous():
                points = points.contiguous()
            if not translations.is_contiguous():
                translations = translations.contiguous()
            if not rotations.is_contiguous():
                rotations = rotations.contiguous()
            points_transformed = torch.empty_like(points)
            _C.transform_points_inverse_forward_cuda(points, translations, rotations, points_transformed)
        else:
            points_transformed = torch.empty_like(points)
            _C.transform_points_inverse_forward(points, translations, rotations, points_transformed)
        ctx.save_for_backward(points, translations, rotations)
        return points_transformed

    @staticmethod
    def backward(ctx, grad_points_transformed):
        points, translations, rotations = ctx.saved_tensors
        if points.is_cuda:
            if not grad_points_transformed.is_contiguous():
                grad_points_transformed = grad_points_transformed.contiguous()
            grad_points = torch.empty_like(points)
            grad_translations = torch.empty_like(translations)
            grad_rotations = torch.empty_like(rotations)
            _C.transform_points_inverse_backward_cuda(grad_points_transformed, points, translations, rotations, grad_points, grad_translations, grad_rotations)
        else:
            grad_points = torch.empty_like(points)
            grad_translations = torch.empty_like(translations)
            grad_rotations = torch.empty_like(rotations)
            _C.transform_points_inverse_backward(grad_points_transformed, points, translations, rotations, grad_points, grad_translations, grad_rotations)
        return grad_points, grad_translations, grad_rotations


def fixed_transform_points_inverse(points, translations, rotations):
    return _FixedTransformPointsInverse.apply(points, translations, rotations)


class _FixedTransformPointsInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, translations, rotations):
        if points.is_cuda:
            if not points.is_contiguous():
                points = points.contiguous()
            if not translations.is_contiguous():
                translations = translations.contiguous()
            if not rotations.is_contiguous():
                rotations = rotations.contiguous()
            points_transformed = torch.empty_like(points)
            _C.fixed_transform_points_inverse_forward_cuda(points, translations, rotations, points_transformed)
        else:
            points_transformed = torch.empty_like(points)
            _C.fixed_transform_points_inverse_forward(points, translations, rotations, points_transformed)
        ctx.save_for_backward(points, translations, rotations)
        return points_transformed

    @staticmethod
    def backward(ctx, grad_points_transformed):
        points, translations, rotations = ctx.saved_tensors
        if points.is_cuda:
            if not grad_points_transformed.is_contiguous():
                grad_points_transformed = grad_points_transformed.contiguous()
            grad_points = torch.empty_like(points)
            _C.fixed_transform_points_inverse_backward_cuda(grad_points_transformed, points, translations, rotations, grad_points)
        else:
            grad_points = torch.empty_like(points)
            _C.fixed_transform_points_inverse_backward(grad_points_transformed, points, translations, rotations, grad_points)
        return grad_points, None, None
