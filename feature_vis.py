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
import umap
import torch
from utils.visualizer import Visualizer
from matplotlib import pyplot as plt

def hyper_xy(e_input,seed = None):
    hyperbolic_hand = umap.UMAP(output_metric='hyperboloid', random_state=seed, n_neighbors=6).fit(e_input.detach().cpu().numpy())
    x = hyperbolic_hand.embedding_[:, 0]
    y = hyperbolic_hand.embedding_[:, 1]
    z = np.sqrt(1 + np.sum(hyperbolic_hand.embedding_ ** 2, axis=1))
    disk_x = x / (1 + z)
    disk_y = y / (1 + z)
    return disk_x, disk_y

@torch.no_grad()
def train_main_worker(opt, model, test_dl, visualizer, device):
    if get_rank() == 0:
        cprint('[*] Start training. name: %s' % opt.name, 'blue')

    #test_dg = get_data_generator(test_dl)
    model.switch_eval()
    # get n_epochs here
    # opt.total_iters = 100000000
    # pbar = tqdm(range(opt.total_iters))
    pbar = tqdm(total=len(test_dl.dataset) //opt.batch_size )
    for i,data in enumerate(test_dl):

        if get_rank() == 0:
            visualizer.reset()

        model.set_input(data)
        if get_rank() == 0:
            f1, f2, f3 = model.forward_eval()
            # f1_x, f1_y = hyper_xy(f1, seed=6)
            # f2_x, f2_y = hyper_xy(f2,seed=6)
            # f3_x, f3_y = hyper_xy(f3,seed=6)
            f_x, f_y =hyper_xy(torch.cat([f1,f2,f3]), seed=24)

            #vis
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            # ax.scatter(f1_x, f1_y, c="blue", cmap='Spectral')
            # ax.scatter(f2_x, f2_y, c="green", cmap='Spectral')
            # ax.scatter(f3_x, f3_y, c="yellow", cmap='Spectral')
            color = ["blue","blue","blue","blue","blue","blue",
                     "green","green","green","green","green","green",
                     "yellow","yellow","yellow","yellow","yellow","yellow"]
            # color = ["blue",
            #          "green",
            #          "yellow"]
            ax.scatter(f_x, f_y, c=color, cmap='Spectral')
            boundary = plt.Circle((0., 0.), 1, fc='none', ec='k')
            ax.add_artist(boundary)
            ax.axis('off');
            #
            plt.show()
        pbar.update(1)


if __name__ == "__main__":
    manualSeed = 111
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    # this will parse args, setup log_dirs, multi-gpus
    opt = TrainOptions().parse_and_setup()
    device = opt.device
    rank = opt.rank

    # CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
    # import pdb; pdb.set_trace()

    # get current time, print at terminal. easier to track exp
    from datetime import datetime

    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')
    shuffer = False
    test_dl_for_eval = CreateTrainEvalDataLoader(opt)
    eval_dl = test_dl_for_eval.dataset

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

    train_main_worker(opt, model, test_dl_for_eval, visualizer, device)