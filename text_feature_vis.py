import os
import time
import inspect
import numpy as np
from termcolor import colored, cprint
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
import utils.util
from options.train_options import TrainOptions
from datasets.dataloader import CreateTrainEvalDataLoader
from models.base_model import create_model
from torchmetrics.image.fid import FrechetInceptionDistance
from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import torch
from utils.visualizer import Visualizer
import umap
import matplotlib
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

@torch.no_grad()
def train_main_worker(opt, model, test_dl, visualizer, device):

    #test_dg = get_data_generator(test_dl)
    model.switch_eval()
    # get n_epochs here
    # opt.total_iters = 100000000
    # pbar = tqdm(range(opt.total_iters))
    pbar = tqdm(total=len(test_dl.dataset) //opt.batch_size )
    out_f_list = []
    cmap = plt.get_cmap('Blues')
    colors = cmap(torch.linspace(0, 1, 20))
    c_list = []
    i = 0
    inde_list = []
    for i,data in enumerate(test_dl):
        #wooden rocking chair
        #a brown wooden rocking chair with no arm rests
        #Wooden rocking chair with 2 wooden legs and bamboo back support wood chair
        #
        target = ["wooden rocking chair",
        "a brown wooden rocking chair with no arm rests",
        "Bright red rocking chair with red geometric patterned fabric and no arms."]
        model.set_input(data)
        for text_item in data['text']:
            if text_item in target:
                if text_item.split(" ")[0] == "wooden":
                    c_list.append([192/255., 0., 0., 1.])
                    inde_list.append(i)
                if text_item.split(" ")[0] == "a":
                    c_list.append([191/ 255., 23/255., 189/255., 1.])
                    inde_list.append(i)
                if text_item.split(" ")[0] == "Bright":
                    c_list.append([197/255., 90/255., 17/255., 1.])
                    inde_list.append(i)
                    print("yes !!!!")

            elif len(text_item.split(" "))>=20:
                c_list.append(colors[20 - 1])
            else:
                c_list.append(colors[len(text_item.split(" "))-1])
        if get_rank() == 0:
            text_f = model.inference(data, infer_all=True)  ##b, 77, 1280
            text_f = torch.mean(text_f, dim=1)
            #text_f,_ = torch.max(text_f, dim=1)
            out_f_list.append(text_f.detach().cpu())

        pbar.update(1)
    out_f_list = torch.cat(out_f_list,dim=0) #1000, 1280
    #vis feature in umap
    mapper = umap.UMAP(output_metric='hyperboloid',random_state=16,n_jobs=1,
                       min_dist=0.5,spread=0.7,n_neighbors=10).fit(out_f_list)
    x = mapper.embedding_[:, 0]
    y = mapper.embedding_[:, 1]
    z = np.sqrt(1 + np.sum(mapper.embedding_ ** 2, axis=1))
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    boundary = plt.Circle((0,0), 1,ec='k', color="white", alpha=0.05)
    ax.add_artist(boundary)
    boundary = plt.Circle((0,0), 1, ec='k', fc='none')
    ax.add_artist(boundary),
    ax.scatter(disk_x, disk_y,c=c_list, cmap='Blues', alpha=1)
    specil_list_x = [disk_x[inde_list[0]],disk_x[inde_list[1]],disk_x[inde_list[2]]]
    specil_list_y = [disk_y[inde_list[0]],disk_y[inde_list[1]],disk_y[inde_list[2]]]
    c_spei = [[192/255., 0., 0., 1.],[191/ 255., 23/255., 189/255., 1.],[197/255., 90/255., 17/255., 1.]]
    ax.scatter(specil_list_x, specil_list_y, c=c_spei, alpha=1)
    ax.axis('off');
    norm = matplotlib.colors.Normalize(1, 77)
    im = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    #cb = plt.colorbar(im, fraction=0.1)
    #cb.ax.tick_params(axis="both", labelsize=10)
    #cb.ax.set_aspect(5)
    plt.show()


if __name__ == "__main__":
    manualSeed = 111
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    # this will parse args, setup log_dirs, multi-gpus
    opt = TrainOptions().parse_and_setup()
    device = opt.device

    # CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
    # import pdb; pdb.set_trace()

    # get current time, print at terminal. easier to track exp
    from datetime import datetime

    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')

    trian_for_vis = CreateTrainEvalDataLoader(opt,number=1000,shuffle=True)
    eval_dl = trian_for_vis.dataset

    dataset_size = len(eval_dl)
    cprint('[*] # training text snippets = %d' % len(eval_dl), 'yellow')

    # main loop
    opt.ckpt = "saved_ckpt/df_steps-81000.pth"
    model = create_model(opt)
    cprint(f'[*] "{opt.model}" initialized.', 'cyan')

    # visualizer
    visualizer = Visualizer(opt)
    if get_rank() == 0:
        visualizer.setup_io()

    # save model and dataset files
    if get_rank() == 0:
        expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
        model_f = inspect.getfile(model.__class__)
        dset_f = inspect.getfile(eval_dl.__class__)
        cprint(f'[*] saving model and dataset files: {model_f}, {dset_f}', 'blue')
        modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
        dsetf_out = os.path.join(expr_dir, os.path.basename(dset_f))
        os.system(f'cp {model_f} {modelf_out}')
        os.system(f'cp {dset_f} {dsetf_out}')

        if opt.vq_cfg is not None:
            vq_cfg = opt.vq_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
            os.system(f'cp {vq_cfg} {cfg_out}')

        if opt.df_cfg is not None:
            df_cfg = opt.df_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(df_cfg))
            os.system(f'cp {df_cfg} {cfg_out}')

    train_main_worker(opt, model, trian_for_vis, visualizer, device)