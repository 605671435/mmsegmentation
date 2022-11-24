# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Sequence
from PIL import Image
import SimpleITK as sitk
import os
import shutil
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
from collections import OrderedDict
def arr_to_img(arr):
    min = np.amin(arr)
    max = np.amax(arr)
    new = (arr - min) * (1. / (max - min)) * 255
    new = new.astype(np.uint8)
    # img = Image.fromarray(new)
    return np.rot90(new)

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
                 draw_ct: bool = False,
                 draw: bool = False,
                 save_featmap = False,
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
        self.draw_ct = draw_ct
        self.save_featmap = save_featmap

        if save_featmap is True:
            if not os.path.exists('./work_dirs/featmaps/tmp'):
                os.makedirs('./work_dirs/featmaps/tmp')
        assert draw_ct is not True or draw_table is not True, \
        'assert error'
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
        if self.save_featmap is True:
            if self.every_n_inner_iters(runner.iter, self.interval):
                os.rename('./work_dirs/featmaps/tmp', './work_dirs/featmaps/iter_' + str(runner.iter))
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

        return dice, acc

    @master_only
    def before_run(self, runner) -> None:
        # self._visualizer.add_config(runner.cfg)
        if self.draw_table is True or self.draw_ct is True:

            vis_backend = runner.visualizer._vis_backends.get('WandbVisBackend')
            self.wandb = vis_backend.experiment

            if hasattr(runner.train_dataloader.dataset, 'METAINFO'):
                self.class_info = runner.train_dataloader.dataset.METAINFO['classes']
            else:
                self.class_info = runner.train_dataloader.dataset.metainfo['classes']

            self.metrics_class = [[]
                                  for _ in self.class_info]

            self.num_classes = len(self.class_info)

            if self.draw_table is True:
                columns = ["id", "iter", "image", "gt", "pred", "dice", "acc"]
                print('INFO***create a wandb table***INFO')
                self.test_table = self.wandb.Table(columns=columns)

                class_id = list(range(self.num_classes - 1)) + [255]
                self.class_set = self.wandb.Classes([{'name': name, 'id': id}
                                           for name, id in zip(self.class_info, class_id)])
            if self.draw_ct is True:
                self.pred_path = os.path.join(self.wandb.run.dir, 'pred_data')
                if not os.path.exists(self.pred_path):
                    os.makedirs(self.pred_path)
                columns = ["id", "iter", "gt", "pred", "dice", "acc"]
                print('INFO***create a wandb table***INFO')
                self.test_table = self.wandb.Table(columns=columns)

    @master_only
    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: dict,
                       outputs) -> None:
        if self.every_n_inner_iters(batch_idx, self.interval):
            if self.draw_ct is True:
                for output in outputs:
                    img_path = output.img_path.split('/')[-1].split('.')[0]

                    gt_sem_seg_arr = arr_to_img(output.gt_sem_seg_3d.data[0].cpu().permute(1, 2, 0).numpy().astype('uint8'))
                    gt_gif = [Image.fromarray(frame) for frame in gt_sem_seg_arr]
                    gt_gif_filename = os.path.join(runner.work_dir, 'gt_tmp.gif')
                    gt_gif[0].save(gt_gif_filename, save_all=True, append_images=gt_gif, duration=0.1)

                    pred_sem_seg = output.pred_sem_seg_3d.data[0].cpu().numpy().astype('uint8')
                    pred_nii_file = sitk.GetImageFromArray(pred_sem_seg)
                    sitk.WriteImage(pred_nii_file, os.path.join(self.pred_path, (img_path + '.nii.gz')))

                    pred_sem_seg_arr = arr_to_img(pred_sem_seg.transpose(1, 2, 0))
                    pred_gif = [Image.fromarray(frame) for frame in pred_sem_seg_arr]
                    pred_gif_filename = os.path.join(runner.work_dir, 'pred_tmp.gif')
                    pred_gif[0].save(pred_gif_filename, save_all=True, append_images=pred_gif, duration=0.1)

                    dice, acc = self.compute_metric(outputs[0].pred_sem_seg_3d.data, outputs[0].gt_sem_seg_3d.data,
                                                    self.num_classes)
                    for i, _ in enumerate(self.class_info):
                        self.metrics_class[i].append(dice[i])   #type: ignore
                    dice = np.round(dice.numpy() * 100, 2)
                    acc = np.round(acc.numpy() * 100, 2)

                    dices = []
                    accs = []
                    form = "{}:{:.2f}, "
                    metrics_pos = ""
                    for i in range(self.num_classes):
                        dices.append(self.class_info[i])
                        dices.append(dice[i])
                        accs.append(self.class_info[i])
                        accs.append(acc[i])
                        metrics_pos += form
                    metrics_pos = "".join(list(metrics_pos)[:-2])
                    print('INFO***add data to table***INFO')
                    self.test_table.add_data(img_path,
                                             runner.iter,
                                             self.wandb.Video(gt_gif_filename),
                                             self.wandb.Video(pred_gif_filename),
                                             metrics_pos.format(*dices),
                                             metrics_pos.format(*accs))
            elif self.draw_table is True:

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

                    dice, acc = self.compute_metric(outputs[0].pred_sem_seg.data, outputs[0].gt_sem_seg.data, self.num_classes)
                    for i, _ in enumerate(self.class_info):
                        self.metrics_class[i].append(dice[i])   #type: ignore
                    dice = np.round(dice.numpy() * 100, 2)
                    acc = np.round(acc.numpy() * 100, 2)

                    annotated = self.wandb.Image(ori_img, classes=self.class_set,
                                            masks={"ground_truth": {"mask_data": gt_sem_seg}})
                    predicted = self.wandb.Image(ori_img, classes=self.class_set,
                                            masks={"predicted_mask": {"mask_data": pred_sem_seg}})
                    print('INFO***add data to table***INFO')

                    self.test_table.add_data(img_path,
                                             runner.iter,
                                             self.wandb.Image(ori_img),
                                             annotated,
                                             predicted,
                                             "{}:{:.2f}, {}:{:.2f}".format(self.class_info[0], dice[0], self.class_info[1], dice[1]),
                                             "{}:{:.2f}, {}:{:.2f}".format(self.class_info[0], acc[0], self.class_info[1], acc[1]))

    @master_only
    def after_val_epoch(self,
                        runner,
                        metrics) -> None:
        metrics_class = OrderedDict({
            f'Dice ({class_key})': np.round(np.nanmean(metric_value) * 100, 2)
            for class_key, metric_value in zip(self.class_info, self.metrics_class)
        })
        self.wandb.log(metrics_class)

    @master_only
    def after_run(self, runner) -> None:
        if self.draw_table is True or self.draw_ct is True:
            print('INFO***log table to wandb***INFO')
            self.wandb.log({"test_predictions": self.test_table})
