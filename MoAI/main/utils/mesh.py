import os
import torch
import numpy as np
import open3d as o3d
import trimesh
from torchvision.utils import save_image
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import matplotlib.pyplot as plt


def render_mesh_rgb(tmesh, intrins, extrins, width, height):
    # Convert tmesh to Open3D legacy mesh
    legacy_mesh = tmesh.to_legacy()

    import pdb; pdb.set_trace()

    # Create an offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # Set up a material for rendering
    material = o3d.visualization.rendering.MaterialRecord()
    # material.shader = "defaultLit"  # Use default lighting shader
    material.shader = "defaultUnlit" 

    # Add the mesh to the scene
    renderer.scene.add_geometry("mesh", legacy_mesh, material)

    # Set the camera projection using intrinsics
    fx, fy = intrins[0, 0], intrins[1, 1]
    cx, cy = intrins[0, 2], intrins[1, 2]
    renderer.scene.camera.set_projection(o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))

    # Set the camera view using extrinsics
    renderer.scene.camera.look_at(
        center=(extrins[:3, 3]),  # Camera target (translation vector from extrinsics)
        eye=(extrins[:3, 3] - extrins[:3, 2]),  # Camera eye position (back-project along z-axis)
        up=(extrins[:3, 1])  # Up vector (usually y-axis direction in extrinsics)
    )

    # Render to an RGB image
    rgb_image = renderer.render_to_image()

    import pdb; pdb.set_trace()
    rgb_image = np.asarray(rgb_image)  # Convert to numpy array

    return rgb_image


def render_depth(
    intrins:o3d.core.Tensor,
    width:int,
    height:int,
    extrins:o3d.core.Tensor,
    tmesh:o3d.t.geometry.TriangleMesh,
    o3d_device:o3d.core.Device
)->np.ndarray:
    """
    Render depth from mesh file

    Parameters
    ----------
    intrins : o3d.core.Tensor
        Camera Intrinsics matrix K: 3x3
    width : int
        image width
    height : int
        image height
    extrins : o3d.core.Tensor
        camera extrinsics matrix 4x4
    tmesh : o3d.t.geometry.TriangleMesh
        TriangleMesh

    Returns
    -------
    np.ndarray
        Rendred depth image
    """

    tmesh = tmesh.to(o3d.core.Device("CPU:0"))

    scene = o3d.t.geometry.RaycastingScene()

    scene.add_triangles(tmesh)
    
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=intrins,
        extrinsic_matrix=extrins,
        width_px=width, height_px=height
    )
    
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    normals = ans["primitive_normals"].numpy()

    return t_hit, normals

def mesh_rendering(mesh, focal_length, extrens_, o3d_device):

    focal_length = float(focal_length.detach().cpu().numpy())
    mesh.compute_vertex_normals()
    
    # camera_info[k].reshape(3, 3)
    intrins_ = np.array([
            [focal_length, 0.0, 256],
            [0.0, focal_length, 256],
            [0.0, 0.0, 1.0]
            ])
    
    width_  = 512 # camera_info.width
    height_ = 512 # camera_info.height

    intrins_t = o3d.core.Tensor(intrins_, device=o3d_device)
    extrins_t = o3d.core.Tensor(extrens_, device=o3d_device)
    
    rendered_depth, normals = render_depth(
        intrins=intrins_t, 
        width=width_, 
        height=height_, 
        extrins = extrins_t, 
        tmesh=mesh,
        o3d_device=o3d_device
    )

    depth_image = torch.tensor(rendered_depth).nan_to_num(posinf=0.0)

    return depth_image, torch.tensor(normals)


def features_to_world_space_mesh(world_space_points, colors, mask=None, edge_threshold=-1, min_triangles_connected=-1, H=512):
    """
    project features to mesh in world space and return (vertices, faces, colors) result by applying simple triangulation from image-structure.

    :param colors: (C, H, W)
    :param depth: (H, W)
    :param fov_in_degrees: fov_in_degrees
    :param world_to_cam: (4, 4)
    :param mask: (H, W)
    :param edge_threshold: only save faces that no any edge are larger than others. Default: -1 (:= do not apply threshold)
    """

    # get point cloud from depth map
    # C, H, W = colors.shape
    # colors = colors.reshape(C, -1)
    # world_space_points = world_space_points.reshape(3, -1)

    C, H, W = colors.shape
    colors = colors.reshape(C, -1)
    world_space_points = world_space_points.reshape(3, -1)

    # define vertex_ids for triangulation
    '''
    00---01
    |    |
    10---11
    '''
    vertex_ids = torch.arange(H*W).reshape(H, W).to(colors.device)
    vertex_00 = remapped_vertex_00 = vertex_ids[:H-1, :W-1]
    vertex_01 = remapped_vertex_01 = (remapped_vertex_00 + 1)
    vertex_10 = remapped_vertex_10 = (remapped_vertex_00 + W)
    vertex_11 = remapped_vertex_11 = (remapped_vertex_00 + W + 1)

    if mask is not None:
        def dilate(x, k=3):
            x = torch.nn.functional.conv2d(
                x.float()[None, None, ...],
                torch.ones(1, 1, k, k).to(mask.device),
                padding="same"
            )
            return x.squeeze() > 0

        # need dilated mask for "connecting vertices", e.g. face at the mask-edge connected to next masked-out vertex
        '''
        x---x---o
        | / | / |  
        x---o---o
        '''
        mask_dilated = dilate(mask, k=5)

        # only keep vertices/features for faces that need to be added (i.e. are masked) -- rest of the faces are already present in 3D
        colors = colors[:, mask_dilated.flatten()]
        world_space_points = world_space_points[:, mask_dilated.flatten()]

        # remap vertex id's to shortened list of vertices
        remap = torch.bucketize(vertex_ids, vertex_ids[mask_dilated])
        remap[~mask_dilated] = -1  # mark invalid vertex_ids with -1 --> due to dilation + triangulation, a few faces will contain -1 values --> need to filter them
        remap = remap.flatten()
        mask_dilated = mask_dilated[:H-1, :W-1]
        vertex_00 = vertex_00[mask_dilated]
        vertex_01 = vertex_01[mask_dilated]
        vertex_10 = vertex_10[mask_dilated]
        vertex_11 = vertex_11[mask_dilated]
        remapped_vertex_00 = remap[vertex_00]
        remapped_vertex_01 = remap[vertex_01]
        remapped_vertex_10 = remap[vertex_10]
        remapped_vertex_11 = remap[vertex_11]

    # triangulation: upper-left and lower-right triangles from image structure
    faces_upper_left_triangle = torch.stack(
        [remapped_vertex_00.flatten(), remapped_vertex_10.flatten(), remapped_vertex_01.flatten()],  # counter-clockwise orientation
        dim=0
    )
    faces_lower_right_triangle = torch.stack(
        [remapped_vertex_10.flatten(), remapped_vertex_11.flatten(), remapped_vertex_01.flatten()],  # counter-clockwise orientation
        dim=0
    )

    # filter faces with -1 vertices and combine
    mask_upper_left = torch.all(faces_upper_left_triangle >= 0, dim=0)
    faces_upper_left_triangle = faces_upper_left_triangle[:, mask_upper_left]
    mask_lower_right = torch.all(faces_lower_right_triangle >= 0, dim=0)
    faces_lower_right_triangle = faces_lower_right_triangle[:, mask_lower_right]
    faces = torch.cat([faces_upper_left_triangle, faces_lower_right_triangle], dim=1)

    # clean mesh
    world_space_points, faces, colors = clean_mesh(
        world_space_points,
        faces,
        colors,
        edge_threshold=edge_threshold,
        min_triangles_connected=min_triangles_connected,
        fill_holes=True
    )

    return world_space_points, faces, colors

def edge_threshold_filter(vertices, faces, edge_threshold=0.5):
    """
    Only keep faces where all edges are smaller than edge length multiplied by edge_threshold.
    Will remove stretch artifacts that are caused by inconsistent depth at object borders

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param edge_threshold: maximum length per edge (otherwise removes that face).

    :return: filtered faces
    """

    p0, p1, p2 = vertices[:, faces[0]], vertices[:, faces[1]], vertices[:, faces[2]]
    d01 = torch.linalg.vector_norm(p0 - p1, dim=0)
    d02 = torch.linalg.vector_norm(p0 - p2, dim=0)
    d12 = torch.linalg.vector_norm(p1 - p2, dim=0)

    d_mean = (d01 + d02 + d12) * edge_threshold

    mask_small_edge = (d01 < d_mean) * (d02 < d_mean) * (d12 < d_mean)
    faces = faces[:, mask_small_edge]

    return faces

def clean_mesh(vertices: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor, edge_threshold: float = 0.1, min_triangles_connected: int = -1, fill_holes: bool = True):
    """
    Performs the following steps to clean the mesh:

    1. edge_threshold_filter
    2. remove_duplicated_vertices, remove_duplicated_triangles, remove_degenerate_triangles
    3. remove small connected components
    4. remove_unreferenced_vertices
    5. fill_holes

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param colors: (3, N) torch.Tensor of type torch.float32 in range (0...1) giving RGB colors per vertex
    :param edge_threshold: maximum length per edge (otherwise removes that face). If <=0, will not do this filtering
    :param min_triangles_connected: minimum number of triangles in a connected component (otherwise removes those faces). If <=0, will not do this filtering
    :param fill_holes: If true, will perform trimesh fill_holes step, otherwise not.

    :return: (vertices, faces, colors) tuple as torch.Tensors of similar shape and type
    """
    if edge_threshold > 0:
        # remove long edges
        faces = edge_threshold_filter(vertices, faces, edge_threshold)

    # cleanup via open3d
    mesh = torch_to_o3d_mesh(vertices, faces, colors)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    if min_triangles_connected > 0:
        # remove small components via open3d
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_triangles_connected
        mesh.remove_triangles_by_mask(triangles_to_remove)

    # cleanup via open3d
    mesh.remove_unreferenced_vertices()

    if fill_holes:
        # misc cleanups via trimesh
        mesh = o3d_to_trimesh(mesh)
        mesh.process()
        mesh.fill_holes()

        return trimesh_to_torch(mesh, v=vertices, f=faces, c=colors)
    else:
        return o3d_mesh_to_torch(mesh, v=vertices, f=faces, c=colors)


def torch_to_o3d_mesh(vertices, faces, colors):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.T.cpu().numpy())
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return mesh

def torch_to_o3d_cuda_mesh(vertices, faces, colors, device=None):

    # import pdb; pdb.set_trace()

    device = None

    if device == None:
        o3d_device = o3d.core.Device("CPU:0")  
    elif device.type == 'cuda':
        # If CUDA, use the index to specify the GPU
        o3d_device = o3d.core.Device(f"CUDA:{device.index}")
    elif device.type == 'cpu':
        # If CPU, use Open3D's CPU device
        o3d_device = o3d.core.Device("CPU:0")
    else:
        raise ValueError(f"Unsupported device type: {device.type}")

    mesh = o3d.t.geometry.TriangleMesh(o3d_device)

    mesh.vertex.positions = o3d.core.Tensor(vertices.T.cpu().numpy(), device=o3d_device)
    mesh.triangle.indices = o3d.core.Tensor(faces.T.cpu().numpy(), device=o3d_device)
    mesh.vertex.colors = o3d.core.Tensor(colors.T.cpu().numpy(), device=o3d_device)

    return mesh, o3d_device

def numpy_to_o3d_cuda_mesh(vertices, faces, normals=None, colors=None, device=None):

    # import pdb; pdb.set_trace()

    device = None

    if device == None:
        o3d_device = o3d.core.Device("CPU:0")  
    elif device.type == 'cuda':
        # If CUDA, use the index to specify the GPU
        o3d_device = o3d.core.Device(f"CUDA:{device.index}")
    elif device.type == 'cpu':
        # If CPU, use Open3D's CPU device
        o3d_device = o3d.core.Device("CPU:0")
    else:
        raise ValueError(f"Unsupported device type: {device.type}")

    mesh = o3d.t.geometry.TriangleMesh(o3d_device)

    mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.float32, device=o3d_device)
    mesh.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.float32, device=o3d_device)

    if colors is not None:
        mesh.vertex.colors = o3d.core.Tensor(colors.T.cpu().numpy(), dtype=o3d.core.float32, device=o3d_device)
    
    if normals is not None:
        mesh.vertex.normals = o3d.core.Tensor(normals, dtype=o3d.core.float32, device=o3d_device)
    
    return mesh, o3d_device


def o3d_mesh_to_torch(mesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.triangles)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.vertex_colors)).T
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def torch_to_o3d_pcd(vertices, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.T.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.T.cpu().numpy())
    return pcd


def o3d_pcd_to_torch(pcd, p=None, c=None):
    points = torch.from_numpy(np.asarray(pcd.points)).T
    if p is not None:
        points = points.to(p)
    colors = torch.from_numpy(np.asarray(pcd.colors)).T
    if c is not None:
        colors = colors.to(c)
    return points, colors


def torch_to_trimesh(vertices, faces, colors):
    mesh = trimesh.base.Trimesh(
        vertices=vertices.T.cpu().numpy(),
        faces=faces.T.cpu().numpy(),
        vertex_colors=(colors.T.cpu().numpy() * 255).astype(np.uint8),
        process=False)

    return mesh


def trimesh_to_torch(mesh: trimesh.base.Trimesh, v=None, f=None, c=None):
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).T
    if v is not None:
        vertices = vertices.to(v)
    faces = torch.from_numpy(np.asarray(mesh.faces)).T
    if f is not None:
        faces = faces.to(f)
    colors = torch.from_numpy(np.asarray(mesh.visual.vertex_colors, dtype=float) / 255).T[:3]
    if c is not None:
        colors = colors.to(c)
    return vertices, faces, colors


def o3d_to_trimesh(mesh: o3d.geometry.TriangleMesh):
    return trimesh.base.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=(np.asarray(mesh.vertex_colors).clip(0, 1) * 255).astype(np.uint8),
        process=False)

def save_mesh(vertices, faces, colors, target_path):
    colors = colors[:3, ...]
    mesh = torch_to_o3d_mesh(vertices, faces, colors)
    mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(target_path, mesh, compressed=True, write_vertex_colors=True, print_progress=True)