import os
import os.path as osp
import argparse
import torch
import time
import logging
from torch.utils.data import DataLoader

from estimator.utils.config import load_config, merge_cli_options, DictAction
from estimator.utils import RunnerInfo, setup_env, fix_random_seed
from estimator.models.builder import build_model
from estimator.datasets.builder import build_dataset
from estimator.tester import Tester

import torch.nn as nn

logger = logging.getLogger('patchrefinerv2')


def parse_args():
    parser = argparse.ArgumentParser(description='PatchRefinerV2 Inference')
    parser.add_argument('config', help='YAML config file path')
    parser.add_argument(
        '--work-dir',
        help='the dir to save logs and models',
        default=None)
    parser.add_argument(
        '--test-type',
        type=str,
        default='normal',
        help='evaluation type: normal (benchmark) or general (custom images)')
    parser.add_argument(
        '--ckp-path',
        type=str,
        default='',
        help='checkpoint path (.pth file)')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision')
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='save colored prediction & depth predictions')
    parser.add_argument(
        '--cai-mode',
        type=str,
        default='m1',
        help='tiling mode: m1 (non-overlapping), m2 (shifted), rN (N random patches)')
    parser.add_argument(
        '--process-num',
        type=int, default=4,
        help='batch size for patch inference')
    parser.add_argument(
        '--tag',
        type=str, default='',
        help='tag for log filename')
    parser.add_argument(
        '--select-patch',
        type=int, default=-1)
    parser.add_argument(
        '--gray-scale',
        action='store_true',
        default=False,
        help='use gray-scale color map')
    parser.add_argument(
        '--image-raw-shape',
        nargs='+', default=[2160, 3840])
    parser.add_argument(
        '--patch-split-num',
        nargs='+', default=[4, 4])
    parser.add_argument(
        '--ssi',
        action='store_true',
        default=False,
        help='use scale-shift-invariant evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config values, e.g. general_dataloader.dataset.rgb_image_dir=./images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    image_raw_shape = [int(num) for num in args.image_raw_shape]
    patch_split_num = [int(num) for num in args.patch_split_num]

    # load YAML config
    cfg = load_config(args.config)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        merge_cli_options(cfg, args.cfg_options)

    # work_dir: CLI > config > default from ckp path
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        if '.pth' in args.ckp_path:
            args.work_dir = osp.dirname(args.ckp_path)
        else:
            args.work_dir = osp.join('work_dir', args.ckp_path.split('/')[1])
        cfg.work_dir = args.work_dir

    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.ckp_path = args.ckp_path

    # fix seed
    seed = cfg.get('seed', 5621)
    fix_random_seed(seed)

    # start distributed env
    if cfg.launcher == 'none':
        distributed = False
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
        rank = 0
        world_size = 1
    else:
        distributed = True
        env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
        rank, world_size, timestamp = setup_env(env_cfg, distributed, cfg.launcher)

    # build dataloader
    if args.test_type == 'normal':
        dataloader_config = cfg.val_dataloader
        dataset = build_dataset(cfg.val_dataloader.dataset)
    elif args.test_type == 'general':
        dataloader_config = cfg.general_dataloader
        dataset = build_dataset(cfg.general_dataloader.dataset)
    else:
        dataloader_config = cfg.val_dataloader
        dataset = build_dataset(cfg.val_dataloader.dataset)

    dataset.image_resolution = image_raw_shape

    # log filename
    config_path = args.config
    exp_cfg_filename = config_path.split('/')[-1].split('.')[0]
    ckp_name = args.ckp_path.replace('/', '_').replace('.pth', '')
    dataset_name = dataset.dataset_name
    log_filename = 'eval_{}_{}_{}_{}_{}.log'.format(
        exp_cfg_filename, args.tag, ckp_name, dataset_name, timestamp)

    # setup logging
    log_file = osp.join(cfg.work_dir, log_filename)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Configure root logger so all module loggers (datasets, tester, etc.) output messages
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    # save runner info
    runner_info = RunnerInfo()
    runner_info.config = cfg
    runner_info.logger = logger
    runner_info.rank = rank
    runner_info.distributed = distributed
    runner_info.launcher = cfg.launcher
    runner_info.seed = seed
    runner_info.world_size = world_size
    runner_info.work_dir = cfg.work_dir
    runner_info.timestamp = timestamp
    runner_info.save = args.save
    runner_info.log_filename = log_filename
    runner_info.gray_scale = args.gray_scale

    if runner_info.save:
        os.makedirs(args.work_dir, exist_ok=True)
        runner_info.work_dir = args.work_dir

    # build model
    if cfg.ckp_path == '':
        model = build_model(cfg.model)
    else:
        model = build_model(cfg.model)
        logger.info('Checkpoint Path: {}. Loading from a local file'.format(cfg.ckp_path))
        ckp = torch.load(cfg.ckp_path, map_location='cpu')
        if hasattr(model, 'load_dict'):
            logger.info(str(model.load_dict(ckp['model_state_dict'])))
        else:
            logger.info(str(model.load_state_dict(ckp['model_state_dict'], strict=True)))

    model.eval()

    if runner_info.distributed:
        torch.cuda.set_device(runner_info.rank)
        model.cuda(runner_info.rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[runner_info.rank], output_device=runner_info.rank,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    else:
        model.cuda()

    if runner_info.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        val_sampler = None

    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataloader_config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=val_sampler)

    # build tester
    tester = Tester(
        config=cfg,
        runner_info=runner_info,
        dataloader=val_dataloader,
        model=model,
        ssi=args.ssi)

    tester.run(
        args.cai_mode,
        process_num=args.process_num,
        image_raw_shape=image_raw_shape,
        patch_split_num=patch_split_num,
        auto_cast=cfg.get('auto_cast', False))


if __name__ == '__main__':
    main()
