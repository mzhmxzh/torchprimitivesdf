import os
import sys
import time
import torch
import trimesh as tm
import torchprimitivesdf
import plotly.graph_objects as go
# from torchsdf import index_vertices_by_faces, compute_sdf

os.environ['CUDA_LAUNCH_BLOCKING'] = 'True'


def box_sdf(scale, x):
    q = x.abs() - scale
    distances = -torch.clamp(q, min=0).square().sum(-1) - torch.clamp(q.max(dim=-1).values, max=0)
    return distances

# def time_torchsdf():
#     # calculate box sdf using torchsdf
#     st_torchsdf = time.time()
#     box_mesh = tm.primitives.Box(extents=scale.detach().cpu().numpy() * 2)
#     vertices = torch.tensor(box_mesh.vertices, dtype=torch.float, device=device)
#     faces = torch.tensor(box_mesh.faces, dtype=torch.long, device=device)
#     face_verts = index_vertices_by_faces(vertices, faces)
#     distances_torchsdf, signs, _, _ = compute_sdf(x_torchsdf, face_verts)
#     distances_torchsdf = torch.sqrt(distances_torchsdf + 1e-8)
#     distances_torchsdf = distances_torchsdf * (-signs)
#     en_torchsdf = time.time()
#     print(f'torchsdf: {format(en_torchsdf - st_torchsdf, ".5f")}s')

def time_pytorch():
    # calculate box sdf using analytical method
    st_mine = time.time()
    # distances_mine, closest_points_mine = torchprimitivesdf.pysdf.box_sdf(scale, x_mine)
    distances_mine = box_sdf(scale, x_mine)
    dis_signs_mine = distances_mine < 0
    distances_mine = distances_mine.abs()
    energy_mine = distances_mine.sum()
    energy_mine.backward()
    grad_x_mine = x_mine.grad
    en_mine = time.time()
    print(f'pytorch: {format(en_mine - st_mine, ".5f")}s')

def time_torchprimitivesdf():
    # calculate box sdf using torchprimitive sdf
    st_ans = time.time()
    distances_ans, dis_signs_ans, closest_points_ans = torchprimitivesdf.box_sdf(x_ans, scale)
    energy_ans = distances_ans.sum()
    energy_ans.backward()
    grad_x_ans = x_ans.grad
    en_ans = time.time()
    print(f'torchprimitivesdf: {format(en_ans - st_ans, ".5f")}s')


if __name__ == '__main__':
    device = torch.device('cuda:2')
    torch.manual_seed(0)
    N = 100000000
    print(f'{N} points')
    
    # initialize box and random points
    scale = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float, device=device)
    x_mine = 4 * scale * (torch.rand([N, 3], dtype=torch.float, device=device) - 0.5)
    x_ans = x_mine.clone().detach()
    x_torchsdf = x_mine.clone().detach()
    x_mine.requires_grad_()
    x_ans.requires_grad_()
    x_torchsdf.requires_grad_()

    # time_torchsdf()
    time_pytorch()
    time_torchprimitivesdf()
