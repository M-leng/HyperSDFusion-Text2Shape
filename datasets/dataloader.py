import torch.utils.data

from datasets.base_dataset import CreateDataset
from datasets.base_dataset import data_sampler
from torch.utils.data import random_split
def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

def CreateDataLoader(opt):
    train_dataset, test_dataset = CreateDataset(opt)
    train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
            drop_last=True,
            )

    test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
            drop_last=False,
            )

    test_dl_for_eval = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=max(int(opt.batch_size // 2), 1),
            sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
            drop_last=False,
        )

    return train_dl, test_dl, test_dl_for_eval

def CreateEvalDataLoader(opt):
    _, test_dataset = CreateDataset(opt)
    sample_len = 100 #1000
    sample_eval_data,  _ = random_split(test_dataset,lengths=[sample_len, len(test_dataset)-sample_len],
                                        generator=torch.Generator().manual_seed(111))
    test_dl_for_eval = torch.utils.data.DataLoader(
            sample_eval_data,
            batch_size=opt.batch_size,
            sampler=data_sampler(sample_eval_data, shuffle=False, distributed=opt.distributed),
            drop_last=False,
        )

    return test_dl_for_eval

def CreateTrainEvalDataLoader(opt, number=100, shuffle=False):
    test_dataset, _ = CreateDataset(opt)
    sample_len = number
    sample_eval_data,  _ = random_split(test_dataset,lengths=[sample_len, len(test_dataset)-sample_len],
                                        generator=torch.Generator().manual_seed(111))
    test_dl_for_eval = torch.utils.data.DataLoader(
            sample_eval_data,
            batch_size=opt.batch_size,
            sampler=data_sampler(sample_eval_data, shuffle=False, distributed=opt.distributed),
            drop_last=False,
        )

    return test_dl_for_eval