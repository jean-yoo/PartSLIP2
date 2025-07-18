import os
import torch
import numpy as np
import trimesh
def normalize_pc(pc_file, save_dir, io, device, save_normalized_pc=False):
    pc = io.load_pointcloud(pc_file, device = device)
    xyz = pc.points_padded().reshape(-1,3)
    rgb = pc.features_padded().reshape(-1,3)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / torch.norm(xyz, dim=1, p=2).max().item()
    xyz = xyz.cpu().numpy()
    rgb = rgb.cpu().numpy()
    if save_normalized_pc:
        save_colored_pc(os.path.join(save_dir, "normalized_pc.ply"), xyz, rgb)
    return xyz, rgb

def normalize_pc_from_mesh(pc_file, save_dir, device, save_normalized_pc=False, n_samples=300000):
    mesh = trimesh.load(pc_file, process=False)

    # Sample points from mesh surface
    points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)

    # Get colors (average vertex color for the face)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        # Get per-face vertex colors and average them
        vertex_colors = mesh.visual.vertex_colors[:, :3]
        colors = vertex_colors[mesh.faces[face_indices]].mean(axis=1)
    else:
        # Default gray if no color
        colors = np.ones_like(points) * 127

    # Normalize
    xyz = torch.tensor(points, dtype=torch.float32, device=device)
    rgb = torch.tensor(colors / 255.0, dtype=torch.float32, device=device)

    xyz = xyz - xyz.mean(dim=0)
    xyz = xyz / torch.norm(xyz, dim=1).max()

    # Convert to numpy for return and optional saving
    xyz_np = xyz.cpu().numpy()
    rgb_np = rgb.cpu().numpy()

    if save_normalized_pc:
        save_colored_pc(os.path.join(save_dir, "normalized_pc.ply"), xyz_np, rgb_np)

    return xyz_np, rgb_np

def save_colored_pc(file_name, xyz, rgb):
    n = xyz.shape[0]
    if rgb.max() < 1.1:
        rgb = (rgb * 255).astype(np.uint8)
    f = open(file_name, "w")
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex %d\n" % n)
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
                
    for i in range(n):
        f.write("%f %f %f %d %d %d\n" % (xyz[i][0], xyz[i][1], xyz[i][2], rgb[i][0], rgb[i][1], rgb[i][2]))

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2'] + 1e-6
    assert bb1['y1'] < bb1['y2'] + 1e-6
    assert bb2['x1'] < bb2['x2'] + 1e-6
    assert bb2['y1'] < bb2['y2'] + 1e-6

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou