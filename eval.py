import os
import time
import inspect

from termcolor import colored, cprint
from tqdm import tqdm
import random
import numpy as np
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

from options.train_options import TrainOptions
# from datasets.dataloader import CreateDataLoader, get_data_generator
from datasets.mini_dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import torch
from utils.visualizer import Visualizer


def train_main_worker(opt, model, test_dl, test_dl_for_eval, visualizer, device):
    if get_rank() == 0:
        cprint('[*] Start training. name: %s' % opt.name, 'blue')
    test_dg = get_data_generator(test_dl)

    for i in range(500, 10500, 500):
        ckpt_name = os.path.join(opt.logs_dir, opt.name, 'ckpt',"df_steps-%s.pth"%i)
        model.load_ckpt(ckpt_name)
        visualizer.reset()
        iter_start_time = time.time()
        metrics = model.eval_metrics(test_dl_for_eval, global_step=i)
        # visualizer.print_current_metrics(epoch, metrics, phase='test')
        visualizer.print_current_metrics(i, metrics, phase='test')
        # print(metrics)

        cprint(f'[*] End of steps %d \t Time Taken: %d sec \n%s' %
                   (
                       i,
                       time.time() - iter_start_time,
                       os.path.abspath(os.path.join(opt.logs_dir, opt.name))
                   ), 'blue', attrs=['bold']
                   )


if __name__ == "__main__":
    # this will parse args, setup log_dirs, multi-gpus
    manualSeed = 111
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    opt = TrainOptions().parse_and_setup()
    device = opt.device
    rank = opt.rank

    # CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])
    # import pdb; pdb.set_trace()

    # get current time, print at terminal. easier to track exp
    from datetime import datetime

    opt.exp_time = datetime.now().strftime('%Y-%m-%dT%H-%M')

    _, test_dl, test_dl_for_eval = CreateDataLoader(opt)
    test_ds = test_dl.dataset

    if opt.dataset_mode == 'shapenet_lang':
        cprint('[*] # testing text snippets = %d' % len(test_ds), 'yellow')
    else:
        cprint('[*] # testing images = %d' % len(test_ds), 'yellow')

    # main loop
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
        cprint(f'[*] saving model and dataset files: {model_f}', 'blue')
        modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
        os.system(f'cp {model_f} {modelf_out}')

        if opt.vq_cfg is not None:
            vq_cfg = opt.vq_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
            os.system(f'cp {vq_cfg} {cfg_out}')

        if opt.df_cfg is not None:
            df_cfg = opt.df_cfg
            cfg_out = os.path.join(expr_dir, os.path.basename(df_cfg))
            os.system(f'cp {df_cfg} {cfg_out}')

    train_main_worker(opt, model, test_dl, test_dl_for_eval, visualizer, device)
