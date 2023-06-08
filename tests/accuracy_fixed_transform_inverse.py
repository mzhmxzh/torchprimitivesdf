import torch
import torchprimitivesdf


def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out


if __name__ == '__main__':
    device = torch.device('cuda:1')
    # device = torch.device('cpu')
    torch.manual_seed(0)
    N = 10000 * 3000
    
    # initialize random points and transformation
    points = torch.rand([N, 3], dtype=torch.float, device=device)
    rotations = robust_compute_rotation_matrix_from_ortho6d(torch.rand([1, 6], dtype=torch.float, device=device))[0]
    translations = torch.rand([3], dtype=torch.float, device=device)
    points.requires_grad_()
    
    # transform points inverse using pytorch
    points_transformed_pytorch = (points - translations) @ rotations
    # points_transformed_pytorch = torchprimitivesdf.fixed_transform_points_inverse(points.cpu(), translations.cpu(), rotations.cpu())
    (points_transformed_pytorch.sum()).backward()
    grad_points_pytorch = points.grad.clone()
    points.grad.data.zero_()
    
    # transform points inverse using torchprimitivesdf
    points_transformed_torchprimitivesdf = torchprimitivesdf.fixed_transform_points_inverse(points, translations, rotations)
    (points_transformed_torchprimitivesdf.sum()).backward()
    grad_points_torchprimitivesdf = points.grad.clone()
    points.grad.data.zero_()
    
    # compare results
    print(f'max points_transformed: {(points_transformed_pytorch.cpu() - points_transformed_torchprimitivesdf.cpu()).abs().max().item()}')
    print(f'max grad_points: {(grad_points_pytorch.cpu() - grad_points_torchprimitivesdf.cpu()).abs().max().item()}')
    print(f'mean points_transformed: {(points_transformed_pytorch.cpu() - points_transformed_torchprimitivesdf.cpu()).abs().mean().item()}')
    print(f'mean grad_points: {(grad_points_pytorch.cpu() - grad_points_torchprimitivesdf.cpu()).abs().mean().item()}')
    assert torch.allclose(points_transformed_torchprimitivesdf.cpu(), points_transformed_pytorch.cpu(), atol=1e-3)
    assert torch.allclose(grad_points_torchprimitivesdf.cpu(), grad_points_pytorch.cpu(), atol=1e-3)
    
    print('success!')
