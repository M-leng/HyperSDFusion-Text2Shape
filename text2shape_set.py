import os
gpu_ids = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"
import pytorch3d

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

from utils.demo_util import SDFusionText2ShapeOpt

seed = 111
opt = SDFusionText2ShapeOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device

# initialize SDFusion model
#ckpt_path = 'saved_ckpt/df_steps_86500.pth'   for chair
ckpt_path = 'saved_ckpt/table-df_steps-latest.pth'
opt.init_model_args(ckpt_path=ckpt_path)

SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')


# txt2shape
out_dir = 'demo_results/ours-table-0131'
if not os.path.exists(out_dir): os.makedirs(out_dir)

# change the input text here to generate different chairs/tables!
input_txt = ["faux wood topped game table on a silver metal pedestal base"]
ddim_steps = 1000
ddim_eta = 0.
uc_scale = 3.
ngen = 1

for itr in range(86500, 90000, 5000):
    if itr!=86500:
        opt.ckpt = f'saved_ckpt/df_steps-{itr}.pth'
        SDFusion.load_ckpt(opt.ckpt)
    for txt in input_txt:
        sdf_gen = SDFusion.txt2shape(input_txt=txt, ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale)
        print(sdf_gen.shape)
        mesh_gen = sdf_to_mesh(sdf_gen)
        # vis as gif
        gen_name = f'{out_dir}/{itr}-txt2shape-{txt}.gif'
        save_mesh_as_gif(SDFusion.renderer, mesh_gen, nrow=3, out_name=gen_name)
        pytorch3d.io.save_obj(f'{out_dir}/ours-{txt}.obj', mesh_gen.verts_list()[0], mesh_gen.faces_list()[0])
        print(f'Input: "{txt}"')
        for name in [gen_name]:
            display(ipy_image(name))