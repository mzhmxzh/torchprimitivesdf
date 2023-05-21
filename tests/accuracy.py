import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(os.path.realpath('.'))

import torch
import trimesh as tm
import torchprimitivesdf
import plotly.graph_objects as go
from torchsdf import index_vertices_by_faces, compute_sdf


if __name__ == '__main__':
    device = torch.device('cuda:1')
    torch.manual_seed(0)
    N = 10000
    
    # initialize box and random points
    scale = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float, device=device)
    x_mine = 4 * scale * (torch.rand([N, 3], dtype=torch.float, device=device) - 0.5)
    x_ans = x_mine.clone()
    x_mine.requires_grad_()
    x_ans.requires_grad_()
    x_ans.retain_grad()
    
    # calculate box sdf using analytical method
    distances_mine, closest_points_mine = torchprimitivesdf.pysdf.box_sdf(scale, x_mine)
    dis_signs_mine = distances_mine < 0
    distances_mine = distances_mine.abs()
    energy_mine = distances_mine.sum()
    energy_mine.backward()
    grad_x_mine = x_mine.grad
    
    # # calculate box sdf using torchsdf
    # box_mesh = tm.primitives.Box(extents=scale.detach().cpu().numpy() * 2)
    # # box_mesh = tm.load('allegro_hand_description/meshes/box.obj', process=False, force='mesh').apply_scale(scale.tolist())
    # vertices = torch.tensor(box_mesh.vertices, dtype=torch.float, device=device)
    # faces = torch.tensor(box_mesh.faces, dtype=torch.long, device=device)
    # face_verts = index_vertices_by_faces(vertices, faces)
    # distances_ans, signs, _, _ = compute_sdf(x, face_verts)
    # distances_ans = torch.sqrt(distances_ans + 1e-8)
    # distances_ans = distances_ans * (-signs)
    
    # calculate box sdf using torchprimitive sdf
    distances_ans, dis_signs_ans, closest_points_ans = torchprimitivesdf.box_sdf(x_ans, scale)
    energy_ans = distances_ans.sum()
    energy_ans.backward()
    grad_x_ans = x_ans.grad
    
    # compare results
    print(f'max dis: {(distances_ans - distances_mine).abs().max().item()}')
    print(f'max closest_points: {(closest_points_ans - closest_points_mine).abs().max().item()}')
    print(f'max grad_x: {(grad_x_ans - grad_x_mine).abs().max().item()}')
    assert torch.isclose(distances_ans, distances_mine).all()
    assert torch.equal(dis_signs_ans, dis_signs_mine)
    assert torch.isclose(closest_points_ans, closest_points_mine, atol=1e-4).all()
    assert torch.isclose(grad_x_ans, grad_x_mine, atol=1e-3).all()
    
    print('success!')
