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
from datasets.dataloader import CreateEvalDataLoader
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
    iou_list = []
    f_score_list = []
    cd_list = []
    fid_list = []
    iter_start_time = time.time()
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for i,data in enumerate(test_dl):

        if get_rank() == 0:
            visualizer.reset()

        model.set_input(data)
        if get_rank() == 0:
            model.inference(data, infer_all=True)
            x = model.x
            x_recon = model.gen_df
            # iou
            iou = utils.util.iou(x, x_recon, 0.)
            iou_list.append(iou.detach().cpu())
            x_mesh = utils.util_3d.sdf_to_mesh(x)
            x_recon_mesh = utils.util_3d.sdf_to_mesh(x_recon)
            f_score, cd = utils.util.calculate_fscore(x_mesh.verts_list(), x_recon_mesh.verts_list(),th=0.01)
            f_score_list.append(f_score)
            cd_list.append(cd)
            model.get_current_visuals()

            imgs_gt= torch.tensor(model.img_gt[:,:3,:,:],dtype=torch.uint8)
            imgs_pr = torch.tensor(model.img_gen_df[:,:3,:,:],dtype=torch.uint8)
            fid = FrechetInceptionDistance(feature=64)
            fid.update(imgs_gt, real=True)
            fid.update(imgs_pr, real=False)
            fid_score = fid.compute()
            del fid
            fid_list.append(fid_score)

        pbar.update(1)

    iou_all = torch.cat(iou_list).mean().numpy()
    cd_all = torch.tensor(cd_list).mean().numpy()
    f_score_all = np.mean(f_score_list)
    fid_all = torch.tensor(fid_list).mean().numpy()
    print(iou_all, cd_all, f_score_all, fid_all)


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

    test_dl_for_eval = CreateEvalDataLoader(opt)
    eval_dl = test_dl_for_eval.dataset

    dataset_size = len(eval_dl)
    cprint('[*] # training text snippets = %d' % len(eval_dl), 'yellow')

    # main loop
    opt.ckpt = "saved_ckpt/df_steps_86500.pth"
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