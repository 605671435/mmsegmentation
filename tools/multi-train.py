# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import copy
import os
import os.path as osp

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.logging import print_log
from mmengine.dist import sync_random_seed
from mmengine.fileio import dump, load
from mmengine.hooks import Hook
from mmengine.runner import Runner, find_latest_checkpoint
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmseg.utils import register_all_modules

EXP_INFO_FILE = 'kfold_exp.json'

prog_description = """K-Fold cross-validation.

To start a 5-fold cross-validation experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5

To resume a 5-fold cross-validation from an interrupted experiment:
    python tools/kfold-cross-valid.py $CONFIG --num-splits 5 --resume
"""  # noqa: E501


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=prog_description)
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--num-splits',
        type=int,
        help='The number of all folds.',
        required=True)
    parser.add_argument(
        '--fold',
        type=int,
        help='The fold used to do validation. '
             'If specify, only do an experiment of the specified fold.')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume the previous experiment.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
             'actual batch size and the original batch size.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def train_single_fold(cfg, num_splits, fold, resume_ckpt=None):
    root_dir = cfg.work_dir
    cfg.work_dir = osp.join(root_dir, f'fold{fold}')
    if resume_ckpt is not None:
        cfg.resume = True
        cfg.load_from = resume_ckpt
    cfg.visualizer.name = f'{cfg.visualizer.name}_fold{str(fold)}'
    cfg.visualizer.vis_backends[1].init_kwargs.name = \
        f'{cfg.visualizer.vis_backends[1].init_kwargs.name}_fold{str(fold)}'

    # build the runner from config
    runner = Runner.from_cfg(cfg)
    runner.logger.info(
        f'----------- Cross-validation: [{fold + 1}/{num_splits}] ----------- ')

    runner.logger.info(f'Train dataset: \n{runner.train_dataloader.dataset}')
    runner.logger.info(f'Val dataset: \n{runner.val_dataloader.dataset}')

    # start training
    runner.train()


def main():
    args = parse_args()

    # register all modules in mmcls into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'


    # # set preprocess configs to model
    # cfg.model.setdefault('data_preprocessor', cfg.get('data_preprocessor', {}))

    # set the unify random seed
    cfg.seed = args.seed or sync_random_seed()

    # resume from the previous experiment
    if args.resume:
        experiment_info = load(osp.join(cfg.work_dir, EXP_INFO_FILE))
        resume_fold = experiment_info['fold']
        cfg.seed = experiment_info['seed']
        resume_ckpt = experiment_info.get('last_ckpt', None)
    else:
        resume_fold = 0
        resume_ckpt = None

    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = range(resume_fold, args.num_splits)

    for fold in folds:
        cfg_ = copy.deepcopy(cfg)
        train_single_fold(cfg_, args.num_splits, fold, resume_ckpt)
        resume_ckpt = None


if __name__ == '__main__':
    main()
