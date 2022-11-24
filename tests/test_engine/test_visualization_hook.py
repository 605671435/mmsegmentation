# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.structures import PixelData

from mmseg.engine.hooks import SegVisualizationHook
from mmseg.structures import SegDataSample, Seg3DDataSample, PixelData3D
from mmseg.visualization import SegLocalVisualizer


class TestVisualizationHook(TestCase):

    # def setUp(self) -> None:
    #
    #     h = 288
    #     w = 512
    #     num_class = 2
    #
    #     SegLocalVisualizer.get_instance('visualizer')
    #     SegLocalVisualizer.dataset_meta = dict(
    #         classes=('background', 'liver', 'abnormal'),
    #         palette=[[120, 120, 120], [6, 230, 230], [255, 255, 255]])
    #
    #     data_sample = SegDataSample()
    #     data_sample.set_metainfo({'img_path': 'tests/data/color.jpg'})
    #     self.data_batch = [{'data_sample': data_sample}] * 2
    #
    #     pred_sem_seg_data = dict(data=torch.randint(0, num_class, (1, h, w)))
    #     pred_sem_seg = PixelData(**pred_sem_seg_data)
    #     pred_seg_data_sample = SegDataSample()
    #     pred_seg_data_sample.set_metainfo({'img_path': 'tests/data/color.jpg'})
    #     pred_seg_data_sample.pred_sem_seg = pred_sem_seg
    #     self.outputs = [pred_seg_data_sample] * 2

    def setUp3d(self) -> None:

        h = 288
        w = 512
        d = 100
        num_class = 3

        SegLocalVisualizer.get_instance('visualizer')
        SegLocalVisualizer.dataset_meta = dict(
            classes=('background', 'liver', 'abnormal'),
            palette=[[120, 120, 120], [6, 230, 230], [255, 255, 255]])

        data_sample = Seg3DDataSample()
        data_sample.set_metainfo({'img_path': './data/biomedical.nii.gz'})
        self.data_batch = [{'inputs': data_sample}] * 2

        pred_sem_seg_data = dict(data=torch.randint(0, num_class, (1, d, h, w)))
        gt_sem_seg_data = dict(data=torch.randint(0, num_class, (1, d, h, w)))
        pred_sem_seg_3d = PixelData3D(**pred_sem_seg_data)
        gt_sem_seg_3d = PixelData3D(**gt_sem_seg_data)
        pred_seg_data_sample = Seg3DDataSample()
        pred_seg_data_sample.set_metainfo({'img_path': './data/biomedical_ann.nii.gz'})
        pred_seg_data_sample.pred_sem_seg_3d = pred_sem_seg_3d
        pred_seg_data_sample.gt_sem_seg_3d = gt_sem_seg_3d
        self.outputs = [pred_seg_data_sample] * 2

    def test_before_run(self):
        runner = Mock()
        runner.iter = 1
        hook = SegVisualizationHook(draw_ct=True, interval=1)
        hook.before_run(runner)

    def test_after_iter(self):
        runner = Mock()
        runner.iter = 1
        hook = SegVisualizationHook(draw_ct=True, interval=1)
        hook.after_iter(
            runner, 1, self.data_batch, self.outputs, mode='train')
        hook.after_iter(runner, 1, self.data_batch, self.outputs, mode='val')
        hook.after_iter(runner, 1, self.data_batch, self.outputs, mode='test')

    def test_after_val_iter(self):
        runner = Mock()
        runner.iter = 2
        # hook = SegVisualizationHook(interval=1)
        # hook.after_val_iter(runner, 1, self.data_batch, self.outputs)
        data_sample = Seg3DDataSample()
        data_sample.set_metainfo({'img_path': './data/biomedical.nii.gz'})
        data_batch = [{'inputs': inputs},
                      {'data_samples': data_samples}] * 2
        hook = SegVisualizationHook(draw_ct=True, interval=1)
        hook.after_val_iter(runner, 1, data_batch, outputs)

        # hook = SegVisualizationHook(
        #     draw=True, interval=1, show=True, wait_time=1)
        # hook.after_val_iter(runner, 1, self.data_batch, self.outputs)

    def test_after_test_iter(self):
        runner = Mock()
        runner.iter = 3
        hook = SegVisualizationHook(draw_ct=True, interval=1)
        hook.after_test_iter(runner, 1, self.data_batch, self.outputs)

    def test_after_run(self):
        runner = Mock()
        runner.iter = 1
        hook = SegVisualizationHook(draw_ct=True, interval=1)
        hook._after_run(runner)

# if __name__ == '__main__':
#     TestVisualizationHook.setUp3d()

