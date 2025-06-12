import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# os.environ["OPEN3D_RENDERING_BACKEND"] = "osmesa"
# os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime"
# if not os.path.exists("/tmp/runtime"):
#     os.makedirs("/tmp/runtime", mode=0o700)
    
import open3d as o3d


# os.environ['PYOPENGL_PLATFORM'] = 'egl'


# ================================================
# Utility: Convert a (N, 6) PyTorch tensor to an Open3D point cloud
# Assumes first 3 columns are XYZ and next 3 columns are RGB in [0,1]
# ================================================
def tensor_to_point_cloud(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    pts = tensor[:, :3]
    colors = tensor[:, 3:6]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# ================================================
# Utility: Compute a "look-at" extrinsic matrix given a camera position,
# target point, and an up direction.
# This produces the view matrix (i.e. world-to-camera transform) used by Open3D.
# ================================================
# def look_at(camera_pos, target, up=np.array([0, 1, 0])):
#     # Compute forward vector (points from target to camera)
#     z = camera_pos - target
#     z = z / np.linalg.norm(z)
#     # Compute right vector
#     x = np.cross(up, z)
#     x = x / np.linalg.norm(x)
#     # Recompute the orthonormal up vector
#     y = np.cross(z, x)
#     R = np.eye(3)
#     R[0, :] = x
#     R[1, :] = y
#     R[2, :] = z
#     t = -R @ camera_pos
#     extrinsic = np.eye(4)
#     extrinsic[:3, :3] = R
#     extrinsic[:3, 3] = t
#     return extrinsic


# def camera_lookat(
#     eye,
#     target,
#     up,
# ):
#     B = eye.shape[0]
#     f = F.normalize(eye - target)
#     l = F.normalize(torch.linalg.cross(up, f))
#     u = F.normalize(torch.linalg.cross(f, l))

#     R = torch.stack((l, u, f), dim=1)  # B 3 3
#     M_R = torch.eye(4, dtype=torch.float32)[None].repeat((B, 1, 1))
#     M_R[..., :3, :3] = R

#     T = -eye
#     M_T = torch.eye(4, dtype=torch.float32)[None].repeat((B, 1, 1))
#     M_T[..., :3, 3] = T

#     return (M_R @ M_T).to(dtype=torch.float32)


def camera_lookat(eye, target, up=np.array([0, 1, 0])):
    """
    Computes a camera-to-world (extrinsic) transformation matrix.
    The transformation matrix M is defined such that:
      p_world = M * p_camera,
    where M = [ R  eye; 0 1 ] with R formed by the camera's right, up, and forward vectors.
    
    Parameters:
        eye (np.array): The camera position in world coordinates (3,).
        target (np.array): The world-space point the camera is looking at (3,).
        up (np.array): The world up vector (default is [0,1,0]).
    
    Returns:
        extrinsic (np.array): A 4x4 camera-to-world transformation matrix.
    """
    # Compute the forward vector (from eye toward target).
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Compute the right vector (perpendicular to both up and forward).
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    
    # Compute the true up vector.
    true_up = np.cross(forward, right)
    
    # Construct the rotation matrix.
    # The columns represent the camera's right, up, and forward directions.
    R = np.column_stack((right, true_up, forward))
    
    # Build the 4x4 extrinsic matrix (camera-to-world).
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = eye  # Last column is the camera (eye) position.
    
    return extrinsic


# ================================================
# Utility: Generate a random camera pose on a sphere of radius 'radius'
# around 'target', with the camera directed towards the target.
# ================================================
def generate_random_camera_pose(target, radius, theta, up=np.array([0,1,0])):
    # theta = np.random.uniform(0, 2 * np.pi)   # azimuth
    phi = np.pi / 9# np.pi / 9 polar angle
    # Spherical to Cartesian conversion (offset from target)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.cos(phi)
    z = radius * np.sin(phi) * np.sin(theta)
    
    # import pdb; pdb.set_trace()
    cam_pos = np.array([x, y, z]) + target
    # extrinsic = look_at(cam_pos, target, up)
    
    extrinsic = camera_lookat(
        cam_pos, # Camera eye position
        target,  # Looking at.
        up,  # z-up
    )
    
    # import pdb; pdb.set_trace()
    
    return extrinsic, np.array([x, z, y, np.rad2deg(phi), np.rad2deg(theta)])

# ================================================
# Utility: Create a camera frustum geometry visualization from an extrinsic.
# This uses Open3D's built-in function to create a LineSet describing a camera.
# ================================================
def create_camera_frustum(extrinsic, intrinsics, width, height, device, color=[1.0, 0.0, 0.0], scale=2.0):
    # Extract intrinsic matrix (3x3 numpy array)
    intrinsic_mat = intrinsics.intrinsic_matrix
        
    intrin_ = o3d.core.Tensor(intrinsic_mat, device=device)
    extrin_ = o3d.core.Tensor(extrinsic, device=device)
    # color = o3d.core.Tensor(color, device=device)    
    # import pdb; pdb.set_trace()
        
    cam_frustum = o3d.t.geometry.LineSet.create_camera_visualization(view_width_px=width, view_height_px=height, intrinsic=intrin_, extrinsic=extrin_, scale=scale, color=color)
    
    return cam_frustum


def extract_eye_center_up(extrinsic: torch.Tensor):
    """
    Given a camera-to-world extrinsic matrix (or a batch of them), returns the camera
    'eye' (position), 'center' (the look-at point computed as eye + forward), and 'up' vector.
    
    Assumes the extrinsic matrix has the form:
        [ R   t ]
        [ 0   1 ]
    where t is the camera position and R's columns are [right, up, forward].
    
    Parameters:
        extrinsic (torch.Tensor): A tensor of shape (4, 4) for a single matrix, or (B, 4, 4)
                                  for a batch.
    
    Returns:
        Tuple of torch.Tensors: (eye, center, up)
          - If extrinsic.shape is (4, 4): returns each of shape (3,)
          - If extrinsic.shape is (B, 4, 4): returns each of shape (B, 3)
    """
    # Single extrinsic matrix case.
    if extrinsic.ndim == 2:
        eye = extrinsic[:3, 3]           # Camera position.
        R = extrinsic[:3, :3]            # Rotation matrix.
        forward = R[:, 2]                # Third column: forward direction.
        center = eye + forward           # Look-at point (1 unit ahead).
        up = R[:, 1]                     # Second column: up vector.
        return eye, center, up, forward
    
    # Batch extrinsic matrix case.
    elif extrinsic.ndim == 3:
        eye = extrinsic[:, :3, 3]        # Shape: (B, 3)
        R = extrinsic[:, :3, :3]         # Shape: (B, 3, 3)
        forward = R[:, :, 2]             # Shape: (B, 3)
        center = eye + forward           # Shape: (B, 3)
        up = R[:, :, 1]                  # Shape: (B, 3)
        return eye, center, up, forward
    
    else:
        raise ValueError("Extrinsic must be of shape (4,4) or (B,4,4)")


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Compute the rotation matrix that rotates vec1 to vec2 using the Rodrigues formula.
    
    Parameters:
        vec1 (numpy.ndarray): Source vector of shape (3,).
        vec2 (numpy.ndarray): Target vector of shape (3,).

    Returns:
        R (numpy.ndarray): A 3x3 rotation matrix that transforms vec1's direction into vec2's direction.
    """
    # Normalize the vectors
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)

    # Compute the cross product and related parameters
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    
    # If the vectors are nearly parallel
    if np.isclose(s, 0):
        # If they are in the same direction, no rotation is required.
        if c > 0:
            return np.eye(3)
        else:
            # Vectors are opposite. Find an orthogonal vector to 'a'.
            orth = np.array([1, 0, 0])
            if np.allclose(a, orth) or np.allclose(a, -orth):
                orth = np.array([0, 1, 0])
            # Create an orthogonal vector and normalize it.
            v = orth - a * np.dot(a, orth)
            v = v / np.linalg.norm(v)
            # The rotation for 180 degrees: R = -I + 2 * outer(v, v)
            return -np.eye(3) + 2 * np.outer(v, v)
    
    # Skew-symmetric cross-product matrix of v.
    K = np.array([[    0, -v[2],  v[1]],
                  [ v[2],     0, -v[0]],
                  [-v[1],  v[0],    0]])
    
    # Rodrigues' formula: R = I + K + K^2 * ((1-c)/s^2)
    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    
    return R

def modify_camera_matrix(cam2world: np.ndarray, N: float, use_old_up: bool = True) -> np.ndarray:
    """
    Given a camera-to-world homogeneous matrix `cam2world`, return a new camera matrix
    whose eye is further from the scene center by a distance N, but still looking at the center.

    The input matrix is assumed to have the form:
        [ R | t ]
        [ 0 | 1 ]
    where:
      - t is the camera position (eye)
      - forward = R[:,2] is the view direction such that center = eye + forward

    Parameters:
        cam2world (np.ndarray): 4x4 input camera-to-world matrix.
        N (float): Additional distance to move the camera eye from the center.
        use_old_up (bool): If True, attempts to use the original up vector from R[:,1].
                           Otherwise, uses the global up [0, 1, 0].

    Returns:
        new_cam2world (np.ndarray): 4x4 camera-to-world matrix with updated camera position.
    """
    # Extract the eye (camera origin) and rotation.
    eye = cam2world[:3, 3]
    R = cam2world[:3, :3]

    # The forward (view) vector is assumed to be the third column of R.
    forward = R[:, 2]
    forward = forward / np.linalg.norm(forward)

    # Assume the scene center is one unit ahead: center = eye + forward
    center = eye + forward

    # Current distance from center.
    d = np.linalg.norm(eye - center)
    if d < 1e-6:
        raise ValueError("The camera eye appears to be at the center. Cannot extend further.")
    
    # New distance: move the eye further away by magnitude N.
    new_d = d + N
    # Move along the vector from center to eye.
    new_eye = center + (eye - center) / d * new_d

    # New forward vector (should point from new_eye to center).
    new_forward = center - new_eye
    new_forward = new_forward / np.linalg.norm(new_forward)

    # Determine the new 'up' vector.
    if use_old_up:
        old_up = R[:, 1]
        new_right = np.cross(old_up, new_forward)
        if np.linalg.norm(new_right) < 1e-6:
            # Fallback to global up if the old up is degenerate.
            global_up = np.array([0.0, 1.0, 0.0])
            new_right = np.cross(global_up, new_forward)
        new_right = new_right / np.linalg.norm(new_right)
        new_up = np.cross(new_forward, new_right)
        new_up = new_up / np.linalg.norm(new_up)
    else:
        # Use the global up vector.
        global_up = np.array([0.0, 1.0, 0.0])
        new_right = np.cross(global_up, new_forward)
        new_right = new_right / np.linalg.norm(new_right)
        new_up = np.cross(new_forward, new_right)
        new_up = new_up / np.linalg.norm(new_up)

    # Assemble new rotation matrix with columns [new_right, new_up, new_forward].
    new_R = np.column_stack((new_right, new_up, new_forward))

    # Assemble the new camera-to-world matrix.
    new_cam2world = np.eye(4)
    new_cam2world[:3, :3] = new_R
    new_cam2world[:3, 3] = new_eye

    return new_cam2world


# ================================================
# Main function
# ================================================
def pointmap_vis(pt_tensor, output_dir, co3d_cam_extrin, co3d_cam_focal, num_cams, radius=1.5):

    # Convert to an Open3D point cloud.
    pcd = tensor_to_point_cloud(pt_tensor)
    
    # Compute the mean of the point coordinates.
    pts = np.asarray(pcd.points)
    mean_pt = pts.mean(axis=0)

    # Create a coordinate frame for reference.
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=mean_pt)

    # -------------------------------
    # Setup camera intrinsic parameters.
    # -------------------------------
    WIDTH = 512
    HEIGHT = 512
    # Here we set arbitrary focal lengths and principal points.
    fx = 525.0
    fy = 525.0
    cx = WIDTH / 2.0
    cy = HEIGHT / 2.0
    
    # device = pt_tensor.device
    # o3d_device = o3d.core.Device(f"CUDA:{device.index}")
    o3d_device = o3d.core.Device("CPU:0")  
    rand_intrinsics = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)

    # -------------------------------
    # Generate random camera poses (and frustums for visualization)
    # -------------------------------
    camera_extrinsics = []
    camera_intrinsics = []
    camera_frustums = []
    
    # sphere1 = o3d.t.geometry.TriangleMesh.create_sphere(radius=0.2)
    # sphere1.translate(mean_pt)
    
    theta = np.linspace(0, 2 * np.pi, 20)
    
    cam_position = []
    
    for i in range(num_cams):
        extrinsic, cam_pos = generate_random_camera_pose(mean_pt, radius, theta[i])
        camera_intrinsics.append(rand_intrinsics)
        # extrinsic = np.linalg.inv(extrinsic)
        camera_extrinsics.append(extrinsic)
        cam_position.append(cam_pos)
        
        # import pdb; pdb.set_trace()
        
        frustum = create_camera_frustum(extrinsic, rand_intrinsics, WIDTH, HEIGHT, o3d_device, color=[0.0, 1.0, 0.0], scale = 0.1)
        camera_frustums.append(frustum)
                    
    for k in range(co3d_cam_extrin.shape[0]):
        
        c_extrin = co3d_cam_extrin[k].detach().cpu().numpy()
        c_extrin = np.linalg.inv(c_extrin)
              
        fx = float(co3d_cam_focal[k])
        fy = fx
        
        c_intrin = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)
        camera_intrinsics.append(c_intrin)
                
        factor = 512 * 100
        # import pdb; pdb.set_trace()
        vis_intrin = o3d.camera.PinholeCameraIntrinsic(int(WIDTH/factor), int(HEIGHT/factor), fx/factor, fy/factor, cx/factor, cy/factor)
        
        frustum = create_camera_frustum(c_extrin, vis_intrin, int(WIDTH/factor), int(HEIGHT/factor), o3d_device, color=[1.0, 0.0, 0.0], scale = 0.1)
        camera_extrinsics.append(c_extrin)
        
        # camera_frustums.append(frustum)
        
        # for i in range(10):
        #     frustum = create_camera_frustum(c_extrin, vis_intrin, WIDTH, HEIGHT, o3d_device, color=[1.0, 0.0, 0.0], scale = 1/(10 * (i+1)))
            # camera_frustums.append(frustum)

    # -------------------------------
    # Prepare the scene: add the point cloud, coordinate frame, and all camera frustums.
    # -------------------------------
    scene_objects = {}  # a dictionary mapping names to geometries
    scene_objects["point_cloud"] = pcd
    # scene_objects["coordinate_frame"] = coord_frame
    for idx, frustum in enumerate(camera_frustums):
        scene_objects[f"camera_{idx}"] = frustum

    # -------------------------------
    # Create output directory if it doesn't exist.
    # -------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -------------------------------
    # Initialize the offscreen renderer.
    # -------------------------------
        
    # Note: Open3D's OffscreenRenderer is available via the open3d.visualization.rendering module.
    renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background

    # Add all scene objects once (assigning default materials)
    default_material = o3d.visualization.rendering.MaterialRecord()
    default_material.shader = "defaultUnlit"
    
    # You can adjust shader settings if needed (e.g., "defaultLit", "defaultUnlit")
    for name, geom in scene_objects.items():
        if "point_cloud" in name:
            # Set point size for the point cloud
            default_material.shader = "defaultUnlit"
        else:
            default_material.shader = "unlitLine"
        renderer.scene.add_geometry(name, geom, default_material)
    
    # default_material.shader = "defaultUnlit"
    # renderer.scene.add_geometry("sphere", sphere1, default_material)

    # -------------------------------
    # For each random camera pose, update the camera view, render, and save the image.
    # -------------------------------
    for i, (extrinsic, intrinsic) in enumerate(zip(camera_extrinsics, camera_intrinsics)):
        # Setup the camera with the generated extrinsic.
        renderer.setup_camera(intrinsic, extrinsic)
        # Render the scene to an image.
        img = renderer.render_to_image()
        # Define output filename.
        out_path = os.path.join(output_dir, f"render_{i:03d}.png")
        o3d.io.write_image(out_path, img)
        print(f"Saved render at {out_path}")
    
    import pdb; pdb.set_trace()