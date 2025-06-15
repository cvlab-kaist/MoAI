import torch
import random
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


# def postprocess_co3d(sample, num_viewpoints = 2, view_setting="combined"):

#     val_list = ["car+106_12658_23657" , "car+194_20901_41098", "motorcycle+365_39123_75802"]

#     if sample["__key__"] in val_list:
#         return None
    
#     instance = sample["__key__"].split("+")[-1]
    
#     a, b, c = instance.split("_")
#     inst= int(a+b+c)

#     try:
#         if view_setting == "random":

#             batch = random.randrange(0, 3)

#             image_list = [obj for obj in sample.keys() if f"b{batch}_image" in obj]

#             idxs = random.sample(range(len(image_list)), num_viewpoints+1)

#             points = sample[f"b{batch}_points.npy"]
#             focals = sample[f"b{batch}_focals.npy"]
#             poses = sample[f"b{batch}_poses.npy"]
#             confs = sample[f"b{batch}_conf.npy"]

#             images = []

#             for i in idxs:
#                 image_key = image_list[i]
#                 images.append(transform(sample[image_key]))
            
#             images = torch.stack(images)
#             output = dict(image = images, points = points[idxs], focals = focals[idxs], pose = poses[idxs], conf = confs[idxs])

#         elif view_setting == "combined":

#             images = []
#             comb_points = []
#             comb_focals = []
#             comb_poses = []
#             comb_confs = []

#             translate_matrix = []

#             for batch in range(3):
#                 image_list = [obj for obj in sample.keys() if f"b{batch}_image" in obj]

#                 b_images = []

#                 for image_key in image_list:
#                     # image_key = image_list[i]
#                     b_images.append(transform(sample[image_key]))
                
#                 if batch == 0:
#                     start_idx = 0
#                     current_poses = sample[f"b{batch}_poses.npy"]
#                     translate_matrix.append(np.eye(4))
#                     current_points = sample[f"b{batch}_points.npy"]

#                 else:
#                     start_idx = 1
#                     last_batch = batch - 1
#                     last_end_pose = sample[f"b{last_batch}_poses.npy"][-1]
#                     cur_begin_pose = sample[f"b{batch}_poses.npy"][0]

#                     this = np.matmul(last_end_pose, np.linalg.inv(cur_begin_pose))
#                     accum = translate_matrix[-1]
#                     translator = np.matmul(accum, this)

#                     translate_matrix.append(translator)
#                     # current_poses = np.matmul(translator, sample[f"b{batch}_poses.npy"].transpose(0, 2, 1)).transpose(0, 2, 1)
#                     current_poses = np.matmul(translator, sample[f"b{batch}_poses.npy"])
#                     current_points = np.matmul(translator[:3,:3], sample[f"b{batch}_points.npy"][...,None]) + translator[:3,3:][None,None,None,...]
#                     current_points = current_points.squeeze(-1)
                    
#                 images.append(torch.stack(b_images)[start_idx:])
#                 comb_points.append(current_points[start_idx:])
#                 comb_focals.append(sample[f"b{batch}_focals.npy"][start_idx:])
#                 comb_confs.append(sample[f"b{batch}_conf.npy"][start_idx:])
#                 comb_poses.append(current_poses[start_idx:])
            
#             points = np.concatenate(comb_points)
#             focals = np.concatenate(comb_focals)
#             poses = np.concatenate(comb_poses)
#             confs = np.concatenate(comb_confs)

#             # idxs = sorted(random.sample(range(points.shape[0]), num_viewpoints+1))

#             # print(idxs)
            
#             start_idx = random.sample(range(5),1)[0]
#             add_idx = random.sample(range(4,9),1)[0]
            
#             idxs = np.linspace(start_idx, start_idx + add_idx, num_viewpoints+1, dtype=int).tolist()
#             # idxs = random.sample(range(len(image_list)), num_viewpoints+1)
#             images = torch.cat(images)
#             # output = dict(image = images, points = points, focals = focals, pose = poses, conf = confs)
#             output = dict(image = images[idxs], points = points[idxs], focals = focals[idxs], pose = poses[idxs], conf = confs[idxs], instances=[inst])

#         return output

#     except:

#         return None

def postprocess_co3d(sample, num_viewpoints = 2, view_setting="combined"):

    val_list = ["car+106_12658_23657" , "car+194_20901_41098", "motorcycle+365_39123_75802"]
    
    transform = ToTensor()

    if sample["__key__"] in val_list:
        return None

    try:
        if view_setting == "random":

            batch = random.randrange(0, 3)

            image_list = [obj for obj in sample.keys() if f"b{batch}_image" in obj]

            idxs = random.sample(range(len(image_list)), num_viewpoints+1)

            points = sample[f"b{batch}_points.npy"]
            focals = sample[f"b{batch}_focals.npy"]
            poses = sample[f"b{batch}_poses.npy"]
            confs = sample[f"b{batch}_conf.npy"]

            images = []

            for i in idxs:
                image_key = image_list[i]
                images.append(transform(sample[image_key]))
            
            images = torch.stack(images)
            output = dict(image = images, points = points[idxs], focals = focals[idxs], pose = poses[idxs], conf = confs[idxs])

        elif view_setting == "combined":

            images = []
            comb_points = []
            comb_focals = []
            comb_poses = []
            comb_confs = []

            translate_matrix = []

            for batch in range(3):
                image_list = [obj for obj in sample.keys() if f"b{batch}_image" in obj]

                b_images = []

                for image_key in image_list:
                    # image_key = image_list[i]
                    b_images.append(transform(sample[image_key]))
                
                if batch == 0:
                    start_idx = 0
                    current_poses = sample[f"b{batch}_poses.npy"]
                    translate_matrix.append(np.eye(4))
                    current_points = sample[f"b{batch}_points.npy"]

                else:
                    start_idx = 1
                    last_batch = batch - 1
                    last_end_pose = sample[f"b{last_batch}_poses.npy"][-1]
                    cur_begin_pose = sample[f"b{batch}_poses.npy"][0]

                    this = np.matmul(last_end_pose, np.linalg.inv(cur_begin_pose))
                    accum = translate_matrix[-1]
                    translator = np.matmul(accum, this)

                    translate_matrix.append(translator)
                    # current_poses = np.matmul(translator, sample[f"b{batch}_poses.npy"].transpose(0, 2, 1)).transpose(0, 2, 1)
                    current_poses = np.matmul(translator, sample[f"b{batch}_poses.npy"])
                    current_points = np.matmul(translator[:3,:3], sample[f"b{batch}_points.npy"][...,None]) + translator[:3,3:][None,None,None,...]
                    current_points = current_points.squeeze(-1)
                    
                images.append(torch.stack(b_images)[start_idx:])
                comb_points.append(current_points[start_idx:])
                comb_focals.append(sample[f"b{batch}_focals.npy"][start_idx:])
                comb_confs.append(sample[f"b{batch}_conf.npy"][start_idx:])
                comb_poses.append(current_poses[start_idx:])
            
            points = np.concatenate(comb_points)
            focals = np.concatenate(comb_focals)
            poses = np.concatenate(comb_poses)
            confs = np.concatenate(comb_confs)

            # idxs = sorted(random.sample(range(points.shape[0]), num_viewpoints+1))

            # print(idxs)
            idxs = random.sample(range(len(image_list)), num_viewpoints+1)
            images = torch.cat(images)
            # output = dict(image = images, points = points, focals = focals, pose = poses, conf = confs)
            output = dict(image = images[idxs], points = points[idxs], focals = focals[idxs], pose = poses[idxs], conf = confs[idxs])

        return output

    except:

        return None
    
    
def postprocess_realestate(sample, num_viewpoints = 2, interpolate_only=False):
    
    transform = ToTensor()

    try:
        image_list = [obj for obj in sample.keys() if "image" in obj]

        idxs = random.sample(range(len(image_list)), num_viewpoints+1)

        if interpolate_only:
            idxs = sorted(idxs)

        points = sample["points.npy"]
        focals = sample["focals.npy"]
        poses = sample["poses.npy"]

        images = []

        for i in idxs:
            image_key = image_list[i]
            images.append(transform(sample[image_key]))

        images = torch.stack(images)
        output = dict(image = images, points = points[idxs], focals = focals[idxs], pose = poses[idxs])

        return output

    except:
        return None
    
def transform_numpy(array):
    """
    Transform a NumPy array image by:
      1. Resizing so the shortest side is 512 pixels (bicubic interpolation).
      2. Center cropping to 512x512.
      3. Normalizing pixel values to [0, 1].
    
    Args:
        image (np.ndarray): Input image in shape (H, W, C) or (H, W).
    
    Returns:
        np.ndarray: Transformed image.
    """
    # Convert numpy array to PIL Image
    # Determine new size keeping the aspect ratio,
    # so that the shortest side becomes 512.
    array = torch.tensor(array).permute(0,3,1,2)
    _, _, height, width = array.shape
    
    if width < height:
        new_width = 512
        new_height = int(512 * height / width)
    else:
        new_height = 512
        new_width = int(512 * width / height)

    # Resize image using bicubic interpolation
    array_resized = F.interpolate(array, size=(new_height, new_width), mode="bilinear")
    # im_resized = im.resize((new_width, new_height), resample=Image.BICUBIC)

    # Compute coordinates for center crop of 512x512
    left = (new_width - 512) // 2
    top = (new_height - 512) // 2
    right = left + 512
    bottom = top + 512

    # Center crop the image
    array = array_resized[...,top:bottom,left:right]

    return array


crop_transform = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),  # Resize shortest side to 512
    transforms.CenterCrop(512),  # Center crop to 512x512
    transforms.ToTensor()    
])

    
def postprocess_vggt(sample, num_viewpoints = 2, interpolate_only=False, view_range=10, inference=False, uniform_sampling=False, sampling_views=10):
      
    if "+" in sample['__key__']:
        dataset = 0 # co3d
        view_range = int(1.0 * view_range)
    else:
        dataset = 1 # realestate

    image_list = [obj for obj in sample.keys() if "frame" in obj]
    image_list = sorted(image_list, key=lambda x : int(x.split("_")[-1][:-4]))
    
    if dataset == 0:
        if len(image_list) <= 10:
            return None
    
    if not inference:
        if not uniform_sampling:
            try:
                range_start_idx = random.sample(range(len(image_list) - view_range - 1),1)[0]
                idxs = random.sample(range(range_start_idx, range_start_idx + view_range), num_viewpoints+1)
            except:
                # range_start_idx = random.sample(range(len(image_list) - view_range - 1),1)[0]
                idxs = random.sample(range(len(image_list)-1), num_viewpoints+1)
                print(f"Image less than {view_range}.")
        else:
            idxs = np.linspace(0, len(image_list)-1, sampling_views, dtype=int).tolist()
    else:
        idxs = [0, 5, 15]

    if interpolate_only:
        idxs = sorted(idxs)
    else:
        random.shuffle(idxs)
        
    try:
        points = sample["pointmap.npy"]
        intrinsic = sample["intrinsic.npy"]
        extrinsic = sample["extrinsic.npy"]
        conf = sample["conf.npy"]

        images = []
        
        for i in idxs:
            image_key = image_list[i]
            images.append(crop_transform(sample[image_key]))
        
        images = torch.stack(images)
        pts = transform_numpy(points[idxs])
        confs = transform_numpy(conf[idxs])
        
        output = dict(image = images, points = pts, intrinsic = intrinsic[idxs], extrinsic = extrinsic[idxs], conf=confs, dataset=[dataset])

        return output
    
    except:
        return None


def postprocess_real_test(sample, num_viewpoints = 2, interpolate_only=False):

    try:
        image_list = [obj for obj in sample.keys() if "image" in obj]

        idxs = random.sample(range(len(image_list)), num_viewpoints+1)

        if interpolate_only:
            idxs = sorted(idxs)

        points = sample["points.npy"]
        focals = sample["focals.npy"]
        poses = sample["poses.npy"]

        images = []

        for i in idxs:
            image_key = image_list[i]
            images.append(transform(sample[image_key]))

        images = torch.stack(images)
        output = dict(image = images, points = points[idxs], focals = focals[idxs], pose = poses[idxs])

        return output

    except:
        return None


def postprocess_combined(sample, num_viewpoints = 2, view_setting="combined", interpolate_only=False, everything_out = False):
    
    transform = ToTensor()

    if "b0_points.npy" in sample.keys() or "b1_points.npy" in sample.keys() or "b2_points.npy" in sample.keys(): # Co3d
        val_list = ["car+106_12658_23657" , "car+194_20901_41098", "motorcycle+365_39123_75802"]

        if sample["__key__"] in val_list:
            return None
        try:
            if view_setting == "random":

                batch = random.randrange(0, 3)

                image_list = [obj for obj in sample.keys() if f"b{batch}_image" in obj]

                idxs = random.sample(range(len(image_list)), num_viewpoints+1)

                points = sample[f"b{batch}_points.npy"]
                focals = sample[f"b{batch}_focals.npy"]
                poses = sample[f"b{batch}_poses.npy"]

                images = []

                for i in idxs:
                    image_key = image_list[i]
                    images.append(transform(sample[image_key]))
                
                images = torch.stack(images)
                output = dict(image = images, points = points[idxs], focals = focals[idxs], pose = poses[idxs])

            elif view_setting == "combined":

                images = []
                comb_points = []
                comb_focals = []
                comb_poses = []

                translate_matrix = []

                for batch in range(3):
                    image_list = [obj for obj in sample.keys() if f"b{batch}_image" in obj]

                    b_images = []

                    for image_key in image_list:
                        # image_key = image_list[i]
                        b_images.append(transform(sample[image_key]))
                    
                    if batch == 0:
                        start_idx = 0
                        current_poses = sample[f"b{batch}_poses.npy"]
                        translate_matrix.append(np.eye(4))
                        current_points = sample[f"b{batch}_points.npy"]

                    else:
                        start_idx = 1
                        last_batch = batch - 1
                        last_end_pose = sample[f"b{last_batch}_poses.npy"][-1]
                        cur_begin_pose = sample[f"b{batch}_poses.npy"][0]

                        this = np.matmul(last_end_pose, np.linalg.inv(cur_begin_pose))
                        accum = translate_matrix[-1]
                        translator = np.matmul(accum, this)

                        translate_matrix.append(translator)
                        # current_poses = np.matmul(translator, sample[f"b{batch}_poses.npy"].transpose(0, 2, 1)).transpose(0, 2, 1)
                        current_poses = np.matmul(translator, sample[f"b{batch}_poses.npy"])
                        current_points = np.matmul(translator[:3,:3], sample[f"b{batch}_points.npy"][...,None]) + translator[:3,3:][None,None,None,...]
                        current_points = current_points.squeeze(-1)
                        
                    images.append(torch.stack(b_images)[start_idx:])
                    comb_points.append(current_points[start_idx:])
                    comb_focals.append(sample[f"b{batch}_focals.npy"][start_idx:])
                    comb_poses.append(current_poses[start_idx:])
                
                points = np.concatenate(comb_points)
                focals = np.concatenate(comb_focals)
                poses = np.concatenate(comb_poses)

                # idxs = sorted(random.sample(range(points.shape[0]), num_viewpoints+1))

                start_idx = random.sample(range(5),1)[0]
                add_idx = random.sample(range(4,9),1)[0]

                if everything_out:
                    idxs = [i for i in range(images[0].shape[0] * len(images)-2)]
                else:
                    idxs = np.linspace(start_idx, start_idx + add_idx, num_viewpoints+1, dtype=int).tolist()
                # idxs = random.sample(range(len(image_list)), num_viewpoints+1)
                images = torch.cat(images)
                # output = dict(image = images, points = points, focals = focals, pose = poses, conf = confs)
                output = dict(image = images[idxs], points = points[idxs], focals = focals[idxs], pose = poses[idxs])

                # print(idxs)
                # idxs = random.sample(range(len(image_list)), num_viewpoints+1)
                # images = torch.cat(images)
                # output = dict(image = images[idxs], points = points[idxs], focals = focals[idxs], pose = poses[idxs])

            return output

        except:
            return None
    
    else: # realestate
        try:
            image_list = [obj for obj in sample.keys() if "image" in obj]

            idxs = random.sample(range(len(image_list)), num_viewpoints+1)

            if interpolate_only:
                idxs = sorted(idxs)

            points = sample["points.npy"]
            focals = sample["focals.npy"]
            poses = sample["poses.npy"]

            images = []

            for i in idxs:
                image_key = image_list[i]
                images.append(transform(sample[image_key]))

            images = torch.stack(images)
            output = dict(image = images, points = points[idxs], focals = focals[idxs], pose = poses[idxs])

            return output

        except:
            return None
    