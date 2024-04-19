import os
gpu_ids = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"


# import libraries
from termcolor import colored, cprint

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from models.base_model import create_model
import matplotlib.pyplot as plt
import torch
import umap
import numpy as np

from utils.demo_util import SDFusionText2ShapeOpt

seed = 111
opt = SDFusionText2ShapeOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device

# initialize SDFusion model
ckpt_path = 'saved_ckpt/df_steps_86500.pth'
opt.init_model_args(ckpt_path=ckpt_path)

SDFusion = create_model(opt)
cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')


# txt2shape
out_dir = 'demo_results'
if not os.path.exists(out_dir): os.makedirs(out_dir)

# change the input text here to generate different chairs/tables!
input_txt = ["a chair",
             "a soft chair",
             "a brown colour soft chair with 4 stand",
             "A new-fashioned, grey soft chair, with two arms and four legs base"]
ngen = 1 # number of generated shapes
ddim_steps = 1000
ddim_eta = 0.
uc_scale = 3.


data_list = []
for itr in range(86500, 90000, 5000):
    if itr!=86500:
        opt.ckpt = f'saved_ckpt/df_steps-{itr}.pth'
        SDFusion.load_ckpt(opt.ckpt)
    for txt in input_txt:
        latent_f = SDFusion.extract_latent(input_txt=txt, ngen=ngen, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale)
        print(latent_f.shape)
        latent_f = latent_f.unsqueeze(0).view(1,-1).cpu()
        print(latent_f.shape)
        data_list.append(latent_f)



# data1 = torch.randn((4096,3))
# data2 = torch.randn((4096,3))
# data3 = torch.randn((4096,3))
data = torch.cat(data_list, dim=0)
data_norm = torch.norm(data,dim=1,keepdim=True)
data = data / data_norm
hyperbolic_maper = umap.UMAP(output_metric='hyperboloid', random_state=0).fit(data)
embedding = hyperbolic_maper.embedding_
x, y = embedding[:, 0], embedding[:, 1]

# 计算 Poincaré 圆盘的坐标
z = np.sqrt(1 + np.sum(embedding ** 2, axis=1))
disk_x = x / (1 + z)
disk_y = y / (1 + z)
# 绘制 Poincaré 圆盘上的点
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
plt.scatter(disk_x[0], disk_y[0], c='r', label='data_1')
plt.scatter(disk_x[1], disk_y[1], c='b', label='data_1')
plt.scatter(disk_x[2], disk_y[2], c='g', label='data_1')
plt.scatter(disk_x[3], disk_y[3], c='y', label='data_1')
boundary = plt.Circle((0, 0), 1, color='black', fill=False)  # 完整的边界圆
ax.add_artist(boundary)
ax.set_aspect('equal', adjustable='box')  # 确保 x 和 y 轴具有相同的比例
ax.axis('off')
plt.show()
