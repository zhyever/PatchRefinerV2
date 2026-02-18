import os
import time
import torch
import torch.distributed as dist
import multiprocessing


def setup_env(env_cfg, distributed, launcher):
    """Setup environment.

    An example of ``env_cfg``::

        env_cfg = dict(
            cudnn_benchmark=True,
            mp_cfg=dict(
                mp_start_method='fork',
                opencv_num_threads=0
            ),
            dist_cfg=dict(backend='nccl', timeout=1800),
            resource_limit=4096
        )

    Args:
        env_cfg (dict): Config for setting environment.
    """
    if env_cfg.get('cudnn_benchmark'):
        torch.backends.cudnn.benchmark = True

    mp_cfg: dict = env_cfg.get('mp_cfg', {})
    mp_start_method = mp_cfg.get('mp_start_method', 'fork')
    opencv_num_threads = mp_cfg.get('opencv_num_threads', 0)
    try:
        multiprocessing.set_start_method(mp_start_method, force=True)
    except RuntimeError:
        pass
    if opencv_num_threads is not None:
        import cv2
        cv2.setNumThreads(opencv_num_threads)

    # init distributed env first, since logger depends on the dist info.
    if distributed and not dist.is_initialized():
        dist_cfg: dict = env_cfg.get('dist_cfg', {})
        backend = dist_cfg.get('backend', 'nccl')
        dist.init_process_group(backend=backend)

    if dist.is_initialized():
        _rank = dist.get_rank()
        _world_size = dist.get_world_size()
    else:
        _rank = 0
        _world_size = 1

    timestamp = torch.tensor(time.time(), dtype=torch.float64, device='cuda')
    # broadcast timestamp from 0 process to other processes
    if dist.is_initialized():
        dist.broadcast(timestamp, src=0)
    _timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
    return _rank, _world_size, _timestamp
