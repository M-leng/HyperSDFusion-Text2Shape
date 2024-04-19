from __future__ import print_function

import os
import random

import numpy as np
from PIL import Image
from einops import rearrange
from chamfer_distance import ChamferDistance as chamfer_dist
import torch
import torchvision.utils as vutils
import open3d
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
import trimesh
from scipy.spatial import cKDTree

################# START: PyTorch Tensor functions #################

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # return image_numpy.astype(imtype)

    n_img = min(image_tensor.shape[0], 16)
    image_tensor = image_tensor[:n_img]

    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    # if image_tensor.shape[1] == 4:
        # import pdb; pdb.set_trace()

    image_tensor = vutils.make_grid( image_tensor, nrow=4 )

    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = ( np.transpose( image_numpy, (1, 2, 0) ) + 1) / 2.0 * 255.
    return image_numpy.astype(imtype)

def tensor_to_pil(tensor):
    # """ assume shape: c h w """
    if tensor.dim() == 4:
        tensor = vutils.make_grid(tensor)

    # assert tensor.dim() == 3
    return Image.fromarray( (rearrange(tensor, 'c h w -> h w c').cpu().numpy() * 255.).astype(np.uint8) )

################# END: PyTorch Tensor functions #################


def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seed_everything(seed):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def normalize_point_clouds(pc):
    centroids = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroids
    m, _ = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def cal_fscore(model, x_gt_list, x_re_list, th=0.01, device=None):
    fscore_list = []
    cd_list = []
    x_gt_f_list = []
    x_re_f_list = []
    for x_gt_item, x_re_item in zip(x_gt_list, x_re_list):
        x_gt_points = trimesh.points.PointCloud(x_gt_item.cpu().detach().numpy())
        x_re_points = trimesh.points.PointCloud(x_re_item.cpu().detach().numpy())
        chd = chamfer_dist()

        if x_gt_item.shape[0] > x_re_item.shape[0]:
            pc_size = x_re_item.shape[0]
            #x_gt_points.vertices = x_gt_points.vertices[np.random.choice(np.arange(len(x_gt_points.vertices)),pc_size)]
            x_gt_points.vertices = farthest_point_sample(x_gt_points.vertices, pc_size)
        elif x_re_item.shape[0] > 2048:
           x_gt_points.vertices = farthest_point_sample(x_gt_points.vertices, 2048)
           x_re_points.vertices = farthest_point_sample(x_re_points.vertices, 2048)
        else:
            pc_size = x_gt_item.shape[0]
            #x_re_points.vertices = x_re_points.vertices[np.random.choice(np.arange(len(x_re_points.vertices)), pc_size)]
            x_re_points.vertices = farthest_point_sample(x_re_points.vertices, pc_size)

        x_gt_points_tensor = torch.tensor(x_gt_points.vertices, dtype=torch.float).unsqueeze(0)
        x_re_points_tensor = torch.tensor(x_re_points.vertices, dtype=torch.float).unsqueeze(0)
        # cd
        dist1, dist2, _, _ = chd(x_gt_points_tensor, x_re_points_tensor)
        cd = (torch.mean(dist1)) + (torch.mean(dist2))

        # fid
        x_gt_points_tensor = x_gt_points_tensor.permute(0, 2, 1)  # b,3,n
        x_re_points_tensor = x_re_points_tensor.permute(0, 2, 1)  # b,3,n
        x_gt_points_tensor = normalize_point_clouds(x_gt_points_tensor)
        x_re_points_tensor = normalize_point_clouds(x_re_points_tensor)
        x_gt_feature = model(x_gt_points_tensor.to(device))
        x_re_feature = model(x_re_points_tensor.to(device))
        x_gt_f_list.append(x_gt_feature)
        x_re_f_list.append(x_re_feature)
        # f_score
        tree1 = cKDTree(x_gt_points.vertices)
        d1, _ = tree1.query(x_re_points.vertices, k=1)

        tree2 = cKDTree(x_re_points.vertices)
        d2, _ = tree2.query(x_gt_points.vertices, k=1)

        if len(d1) and len(d2):
            recall = float(sum(d < th for d in d2)) / float(len(d2))
            precision = float(sum(d < th for d in d1)) / float(len(d1))
            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0

        fscore_list.append(fscore)
        cd_list.append(cd)

    re_fscore = np.mean(fscore_list)
    cd = torch.tensor(cd_list)
    cd = cd.mean()
    x_gt_f = torch.cat(x_gt_f_list, dim=0)
    x_re_f = torch.cat(x_re_f_list, dim=0)
    return re_fscore, cd, x_gt_f, x_re_f


def calculate_fscore(gt, pr, th: float = 0.01):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    f_socre_list = []
    chd = chamfer_dist()
    cd_list = []
    for x_gt_item, x_re_item in zip(gt, pr):
        #sample
        gt_pcd = open3d.geometry.PointCloud()
        pr_pcd = open3d.geometry.PointCloud()
        if x_gt_item.shape[0] > x_re_item.shape[0]:
            pc_size = x_re_item.shape[0]
            # x_gt_points.vertices = x_gt_points.vertices[np.random.choice(np.arange(len(x_gt_points.vertices)),pc_size)]
            gt_pcd.points = open3d.utility.Vector3dVector(farthest_point_sample(x_gt_item.cpu().detach().numpy(), pc_size))
            pr_pcd.points = open3d.utility.Vector3dVector(x_re_item.cpu().detach().numpy())
        else:
            pc_size = x_gt_item.shape[0]
            gt_pcd.points = open3d.utility.Vector3dVector(x_gt_item.cpu().detach().numpy())
            pr_pcd.points = open3d.utility.Vector3dVector(farthest_point_sample(x_re_item.cpu().detach().numpy(), pc_size))

        d1 = gt_pcd.compute_point_cloud_distance(pr_pcd)
        d2 = pr_pcd.compute_point_cloud_distance(gt_pcd)


        if len(d1) and len(d2):
            recall = float(sum(d < th for d in d2)) / float(len(d2))
            precision = float(sum(d < th for d in d1)) / float(len(d1))

            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0
        else:
            fscore = 0
            precision = 0
            recall = 0
        f_socre_list.append(fscore)
        #compute cd
        x_gt_points_tensor = torch.tensor(gt_pcd.points, dtype=torch.float).unsqueeze(0)
        x_re_points_tensor = torch.tensor(pr_pcd.points, dtype=torch.float).unsqueeze(0)
        # cd
        dist1, dist2, _, _ = chd(x_gt_points_tensor, x_re_points_tensor)
        cd = (torch.mean(dist1)) + (torch.mean(dist2))
        cd_list.append(cd)

    return np.mean(f_socre_list), torch.tensor(cd_list).mean()

def calculate_hmd(point_set):
    chd = chamfer_dist()
    hmd_list = []
    for pc_item_a in point_set:
        for pc_item_b in point_set:
            #sample
            a_pcd = open3d.geometry.PointCloud()
            b_pcd = open3d.geometry.PointCloud()
            if pc_item_a.shape[0] > pc_item_b.shape[0]:
                pc_size = pc_item_b.shape[0]
                # x_gt_points.vertices = x_gt_points.vertices[np.random.choice(np.arange(len(x_gt_points.vertices)),pc_size)]
                a_pcd.points = open3d.utility.Vector3dVector(farthest_point_sample(pc_item_a.cpu().detach().numpy(), pc_size))
                b_pcd.points = open3d.utility.Vector3dVector(pc_item_b.cpu().detach().numpy())
            else:
                pc_size = pc_item_a.shape[0]
                a_pcd.points = open3d.utility.Vector3dVector(pc_item_a.cpu().detach().numpy())
                b_pcd.points = open3d.utility.Vector3dVector(farthest_point_sample(pc_item_b.cpu().detach().numpy(), pc_size))
            #compute cd
            a_pcd_tensor = torch.tensor(a_pcd.points, dtype=torch.float).unsqueeze(0)
            b_pcd_tensor = torch.tensor(b_pcd.points, dtype=torch.float).unsqueeze(0)
            # cd
            dist1, dist2, _, _ = chd(a_pcd_tensor, b_pcd_tensor)
            cd = (torch.mean(dist1)) + (torch.mean(dist2))
            print(cd)
            hmd_list.append(cd)
    hdm_tensor = torch.tensor(hmd_list).mean()
    return hdm_tensor

def iou(x_gt, x, thres):
    thres_gt = 0.0

    # compute iou
    # > 0 free space, < 0 occupied
    x_gt_mask = x_gt.clone().detach()
    x_gt_mask[x_gt > thres_gt] = 0.
    x_gt_mask[x_gt <= thres_gt] = 1.

    x_mask = x.clone().detach()
    x_mask[x > thres] = 0.
    x_mask[x <= thres] = 1.

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    inter = rearrange(inter, 'b c d h w -> b (c d h w)')
    union = rearrange(union, 'b c d h w -> b (c d h w)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    return iou

#################### START: MISCELLANEOUS ####################
def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params

#################### END: MISCELLANEOUS ####################



# Noam Learning rate schedule.
# From https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py
class NoamLR(_LRScheduler):
	
	def __init__(self, optimizer, warmup_steps):
		self.warmup_steps = warmup_steps
		super().__init__(optimizer)

	def get_lr(self):
		last_epoch = max(1, self.last_epoch)
		scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
		return [base_lr * scale for base_lr in self.base_lrs]