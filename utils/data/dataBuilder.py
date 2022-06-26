import random
from functools import partial
from torch.utils.data import DataLoader
import numpy as np
import torch
from Code.utils.data import collate
from torch import distributed as dist

# def build_dataset(cfg, default_args=None):
#     """Build datasets."""
#     return


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    """Build PyTorch DataLoader.
    """
    batch_size = samples_per_gpu
    num_workers = workers_per_gpu
    rank, w_size = get_dist_info()
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        **kwargs)
    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
