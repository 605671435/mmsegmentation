# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Sequence

import numpy as np
import mmcv
import torch
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.dist import master_only
from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 draw_table: bool = False,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 file_client_args: dict = dict(backend='disk')):
        self._visualizer: SegLocalVisualizer = \
            SegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.draw = draw
        self.draw_table = draw_table
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    @master_only
    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = self.file_client.get(img_path)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)

    def compute_metric(self, pred_sem_seg, gt_sem_seg, num_classes):
        intersect = pred_sem_seg[pred_sem_seg == gt_sem_seg]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_sem_seg.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            gt_sem_seg.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        # area_union = area_pred_label + area_label - area_intersect

        dice = 2 * area_intersect / (
                area_pred_label + area_label)
        acc = area_intersect / area_label

        return np.round(dice.numpy() * 100, 2), np.round(acc.numpy() * 100, 2)

    @master_only
    def before_run(self, runner) -> None:
        # self._visualizer.add_config(runner.cfg)
        if self.draw_table is True:
            vis_backend = runner.visualizer._vis_backends.get('WandbVisBackend')
            wandb = vis_backend.experiment
            columns = ["id", "iter", "image", "gt", "pred", "dice", "acc"]
            print('INFO***create a wandb table***INFO')
            self.test_table = wandb.Table(columns=columns)
            if hasattr(runner.train_dataloader.dataset, 'METAINFO'):
                classes = runner.train_dataloader.dataset.METAINFO['classes']
            else:
                classes = runner.train_dataloader.dataset.metainfo['classes']
            num_classes = len(classes)
            class_id = list(range(num_classes - 1)) + [255]
            self.class_set = wandb.Classes([{'name': name, 'id': id}
                                       for name, id in zip(classes, class_id)])

    @master_only
    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: dict,
                       outputs) -> None:
        if self.draw_table is True and self.every_n_inner_iters(batch_idx, self.interval):
            vis_backend = runner.visualizer._vis_backends.get('WandbVisBackend')
            wandb = vis_backend.experiment

            classes = runner.train_dataloader.dataset.METAINFO['classes']
            # palette = runner.train_dataloader.dataset.METAINFO['palette']
            num_classes = len(classes)

            for output in outputs:
                img_path = output.img_path.split('/')[-1].split('.')[0]
                ori_img = data_batch['inputs'][0].permute(1, 2, 0).cpu().numpy().astype('uint8')

                gt_sem_seg = output.gt_sem_seg.data.cpu().permute(1, 2, 0).numpy().astype('uint8') * 255
                gt_sem_seg = mmcv.imresize(gt_sem_seg[..., -1],
                                           (ori_img.shape[1], ori_img.shape[0]),
                                           interpolation='nearest',
                                           backend='pillow')
                pred_sem_seg = output.pred_sem_seg.data.cpu().permute(1, 2, 0).numpy().astype('uint8') * 255
                pred_sem_seg = mmcv.imresize(pred_sem_seg[..., -1],
                                             (ori_img.shape[1], ori_img.shape[0]),
                                             interpolation='nearest',
                                             backend='pillow')

                dice, acc = self.compute_metric(outputs[0].pred_sem_seg.data, outputs[0].gt_sem_seg.data, num_classes)

                annotated = wandb.Image(ori_img, classes=self.class_set,
                                        masks={"ground_truth": {"mask_data": gt_sem_seg}})
                predicted = wandb.Image(ori_img, classes=self.class_set,
                                        masks={"predicted_mask": {"mask_data": pred_sem_seg}})
                print('INFO***add data to table***INFO')
                self.test_table.add_data(img_path,
                                         runner.iter,
                                         wandb.Image(ori_img),
                                         annotated,
                                         predicted,
                                         "{}:{}, {}:{}".format(classes[0], dice[0], classes[1], dice[1]),
                                         "{}:{}, {}:{}".format(classes[0], acc[0], classes[1], acc[1]))

    @master_only
    def after_run(self, runner) -> None:
        if self.draw_table is True:
            vis_backend = runner.visualizer._vis_backends.get('WandbVisBackend')
            wandb = vis_backend.experiment
            print('INFO***log table to wandb***INFO')
            wandb.log({"test_predictions": self.test_table})
