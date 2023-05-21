import torch

def box_sdf(scale, x):
    """
    Calculate signed distances from points to box in box frame
    
    Interiors are positive, exteriors are negative
    
    Use analytical method
    
    Parameters
    ----------
    scale: (3,) torch.Tensor
        box scales, [-scale[0], scale[0]] * [-scale[1], scale[1]] * [-scale[2], scale[2]]
    x: (N, 3) torch.Tensor
        points
    
    Returns
    -------
    dis: (N,) torch.Tensor
        signed distances from points to box
    """

    nearest_point = x.unsqueeze(1).repeat(1, 6, 1).detach()  # (N, 6, 3)
    nearest_point[:, 0, 0] = -scale[0]
    nearest_point[:, 1, 0] = scale[0]
    nearest_point[:, 2, 1] = -scale[1]
    nearest_point[:, 3, 1] = scale[1]
    nearest_point[:, 4, 2] = -scale[2]
    nearest_point[:, 5, 2] = scale[2]
    nearest_point[:, 2:6, 0] = torch.clamp(nearest_point[:, 2:6, 0], -scale[0], scale[0])
    nearest_point[:, 0:2, 1] = torch.clamp(nearest_point[:, 0:2, 1], -scale[1], scale[1])
    nearest_point[:, 4:6, 1] = torch.clamp(nearest_point[:, 4:6, 1], -scale[1], scale[1])
    nearest_point[:, 0:4, 2] = torch.clamp(nearest_point[:, 0:4, 2], -scale[2], scale[2])
    
    tmp = (x.unsqueeze(1) - nearest_point).square().sum(-1).min(-1)
    dis = tmp.values  # (N,)
    nearest_point = nearest_point[range(len(nearest_point)), tmp.indices]
    interior = (x[:, 0] >= -scale[0]) & (x[:, 0] <= scale[0]) & (x[:, 1] >= -scale[1]) & (x[:, 1] <= scale[1]) & (x[:, 2] >= -scale[2]) & (x[:, 2] <= scale[2])
    dis = torch.where(interior, dis, -dis)
    
    return dis, nearest_point
    
