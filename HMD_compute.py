import os
gpu_ids = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"


# import libraries
import numpy as np
from IPython.display import Image as ipy_image
from IPython.display import display
from termcolor import colored, cprint

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision.utils as vutils

from models.base_model import create_model
from utils.util_3d import render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif
from utils.util import calculate_hmd
from utils.demo_util import SDFusionText2ShapeOpt

seed = 2023
opt = SDFusionText2ShapeOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device

# initialize SDFusion model
#ckpt_path = 'logs_home/part4-36k-88k/ckpt/df_steps-latest.pth'
ckpt_path = 'saved_ckpt/df_steps_86500.pth'
opt.init_model_args(ckpt_path=ckpt_path)

SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')


# txt2shape
out_dir = 'demo_results'
if not os.path.exists(out_dir): os.makedirs(out_dir)

# change the input text here to generate different chairs/tables!
input_txt = "A chair."

ngen = 6 # number of generated shapes
ddim_steps = 100
ddim_eta = 0.
uc_scale = 3.
times = 1
hmd_total = torch.tensor(0.)
for i in range(times):
    sdf_gen = SDFusion.txt2shape(input_txt=input_txt, ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale)
    mesh_gen = sdf_to_mesh(sdf_gen)
    mesh_gen = mesh_gen.verts_list()
    hmd = calculate_hmd(mesh_gen)
    hmd_total += hmd

print(times, "mean hmd:",hmd_total/times)