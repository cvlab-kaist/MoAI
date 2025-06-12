import torch
import numpy as np
from .mesh import features_to_world_space_mesh, torch_to_o3d_cuda_mesh, mesh_rendering, get_rays

def depth_normalize(min, max, depth):
    t_min = torch.tensor(min, device=depth.device)
    t_max = torch.tensor(max, device=depth.device)

    normalized_depth = (((depth - t_min) / (t_max - t_min)) - 0.5 ) * 2.0

    return normalized_depth

@torch.no_grad()
def embedding_prep(images, correspondence, ref_camera, tgt_camera, src_idx, batch_size, device, current_dataset = None):
    
    mesh_pts = None
    mesh_depth = None
    mesh_normals = None
    mesh_ref_normals = None
    mesh_normal_mask = None
    norm_depth = None
    confidence_map = None
    plucker = None 
    tgt_depth = None
    
    # origins = ref_camera['pose'][:,:,:3,-1]
    tgt_origins = tgt_camera['pose'][:,:3,-1]
    # dist = torch.linalg.norm(correspondence["ref"] - origins[...,None,None,:],axis=-1)[...,None]
    dist = torch.linalg.norm(correspondence["ref"] - tgt_origins[...,None,None,None,:],axis=-1)[...,None]
    norm_depth = dist
    
    # save_image(ref_depth[0].permute(0,3,1,2) * 0.5 + 0.5,"new__3.png")
    # tgt_origins = tgt_camera['pose'][:,:3,-1]
    tgt_dist = torch.linalg.norm(correspondence["tgt"] - tgt_origins[...,None,None,:],axis=-1)[...,None]
    tgt_depth = tgt_dist
                                
    ref_depth = depth_normalize(0.4285, 2.2866, norm_depth)
    tgt_depth_norm = depth_normalize(0.4285, 2.2866, tgt_depth)
    
    clip = True
                                
    if clip:
        min_val = -1.0
        max_val = 1.0
        ref_depth = torch.clip(ref_depth, min=min_val, max=max_val).reshape(-1,518,518,1).permute(0,3,1,2).repeat(1,3,1,1)
        tgt_depth_norm = torch.clip(tgt_depth_norm, min=min_val, max=max_val).permute(0,3,1,2).repeat(1,3,1,1)
    
    # import pdb; pdb.set_trace()
    
    use_mesh = True
    downsample = True
    downsample_by = 2
    
    use_normal = False
    use_normal_mask = False

    if use_mesh:
        if downsample:
            start = downsample_by // 2
            interval = downsample_by
            points = correspondence['ref'][:,:,start::interval,start::interval,:].permute(0,1,4,2,3).float()            
            images_ref = torch.cat(images["ref"], dim=1)
            rgb = images_ref[:,:,:,start::interval,start::interval]
            side_length = 512 // downsample_by

        # else:
        #     points = batch['points'].reshape(batch_size, -1, 3).permute(0,2,1).float()
        #     rgb = batch['image'].permute(0,1,3,4,2).reshape(batch_size, -1, 3).permute(0,2,1)
        #     side_length = 512 // downsample_by

        batch_pts = points
        batch_colors = rgb
        orig_length = torch.tensor(518).to(device)

        mesh_pts = []
        mesh_normals = []
        mesh_depth = []
        mesh_ref_normal_list = []
        mesh_normal_mask = []

        for i, (pts_list, color_list) in enumerate(zip(batch_pts, batch_colors)):
            extrins = tgt_camera["pose"][i].detach().cpu().numpy()
            focal_length = tgt_camera["focals"][i]

            vert = []
            fc = []
            col = []

            vert_stack = 0

            for k, (pts, color) in enumerate(zip(pts_list, color_list)):
                # import pdb; pdb.set_trace()

                vertices, faces, colors = features_to_world_space_mesh(
                    world_space_points=pts.detach(),
                    colors=color.detach(),
                    edge_threshold=0.48,
                    H = side_length
                )

                vert.append(vertices)
                fc.append(faces + vert_stack)
                col.append(colors)

                vert_num = vertices.shape[1]
                vert_stack += vert_num

            vertices = torch.cat(vert, dim=-1)
            faces = torch.cat(fc, dim=-1)
            colors = torch.cat(col, dim=-1)

            mesh, o3d_device = torch_to_o3d_cuda_mesh(vertices, faces, colors, device = pts.device)

            inv_extrins = np.linalg.inv(extrins)
            rendered_depth, normals = mesh_rendering(mesh, focal_length, inv_extrins, o3d_device)
            rays_o, rays_d = get_rays(orig_length, orig_length, focal_length.to(device), torch.tensor(extrins).to(device), 1, device)
            mask = (rendered_depth != 0)

            proj_pts = mask[...,None].to(device) * (rays_o[0,0] + rendered_depth[...,None].to(device) * rays_d[0,0])
            
            if use_normal_mask and current_dataset != "realestate":
                center_dir = rays_d[0, 0, 259, 259][None,None,...] 

                normed_center_dir = -center_dir / torch.norm(center_dir, dim=-1, keepdim=True) 
                normed_normals = normals.to(device) / torch.norm(normals, dim=-1, keepdim=True).to(device)

                dot_product = torch.clamp(torch.sum(normed_center_dir * normed_normals, dim=-1, keepdim=True), -1.0, 1.0) 
                angle_difference = torch.acos(dot_product)

                angle_mask = angle_difference > (torch.pi * 1/2)
                mesh_normal_mask.append(angle_mask)

            mesh_pts.append(proj_pts)
            mesh_depth.append(rendered_depth)
            mesh_normals.append(normals)

            if use_normal:
                per_batch_ref_normals = []

                for ref_extrins, ref_focal in zip(ref_camera["pose"][i], ref_camera["focals"][i]):
                    ref_extrins = ref_extrins.detach().cpu().numpy()
                    ref_pose = np.linalg.inv(ref_extrins)
                    ref_depths, ref_normals = mesh_rendering(mesh, ref_focal, ref_pose, o3d_device)
                    per_batch_ref_normals.append(ref_normals.permute(2,0,1))
                
                mesh_ref_normals = torch.stack(per_batch_ref_normals)
                mesh_ref_normal_list.append(mesh_ref_normals)
            
        mesh_pts = torch.stack(mesh_pts)
        mesh_depth = torch.stack(mesh_depth).to(device)
        mesh_normals = torch.stack(mesh_normals)

        if use_normal:
            mesh_ref_normals = torch.stack(mesh_ref_normal_list)

        if use_normal_mask and current_dataset != "realestate":
            # save_image(torch.cat((mesh_normals.permute(0,3,1,2).to(device), mesh_normal_mask[0][None,...].permute(0,3,1,2).repeat(1,3,1,1))), "new.png")

            mesh_normal_mask = (1 - torch.stack(mesh_normal_mask).float()).to(device)

            mesh_pts = mesh_pts * mesh_normal_mask
            mesh_depth = mesh_depth * mesh_normal_mask[...,0]
            mesh_normals = mesh_normals.to(device) * mesh_normal_mask
            
    return tgt_depth_norm, ref_depth, mesh_pts, mesh_depth, mesh_normals, mesh_ref_normals, mesh_normal_mask, norm_depth, confidence_map, plucker, tgt_depth
