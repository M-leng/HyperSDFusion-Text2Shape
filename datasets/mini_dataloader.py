import torch.utils.data

from datasets.base_dataset import CreateDataset
from datasets.base_dataset import data_sampler
from torch.utils.data import random_split

def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

def CreateDataLoader(opt):
    train_dataset, _ = CreateDataset(opt)
    mini_factor = 0.01
    train_len = int(len(train_dataset) * mini_factor)
    mini_train_data, mini_test_data, _ = random_split(train_dataset,
                                                     lengths=[int(train_len * 0.8),
                                                              int(train_len * 0.2),
                                                              len(train_dataset) - int(train_len * 0.8) - int(
                                                                  train_len * 0.2)], )
    train_dl = torch.utils.data.DataLoader(
            mini_train_data,
            batch_size=opt.batch_size,
            sampler=data_sampler(mini_train_data, shuffle=True, distributed=opt.distributed),
            drop_last=True,
            )

    test_dl = torch.utils.data.DataLoader(
            mini_test_data,
            batch_size=opt.batch_size,
            sampler=data_sampler(mini_test_data, shuffle=False, distributed=opt.distributed),
            drop_last=False,
            )

    test_dl_for_eval = torch.utils.data.DataLoader(
            mini_test_data,
            batch_size=max(int(opt.batch_size // 2), 1),
            sampler=data_sampler(mini_test_data, shuffle=False, distributed=opt.distributed),
            drop_last=False,
        )

    return train_dl, test_dl, test_dl_for_eval
