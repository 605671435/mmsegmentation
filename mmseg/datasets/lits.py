# Copyright (c) OpenMMLab. All rights reserved.

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LITSDataset(BaseSegDataset):
    """ChestXray dataset.

    In segmentation map annotation for ChestXray, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    # METAINFO = dict(
    #     classes=('background', 'liver', 'abnormal'),
    #     palette=[[120, 120, 120], [6, 230, 230],  [255, 255, 255]])
    METAINFO = dict(
        classes=('background', 'liver'),
        palette=[[120, 120, 120], [6, 230, 230]])
    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.nii.gz',
            seg_map_suffix='.nii.gz',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.data_prefix['img_path'])

@DATASETS.register_module()
class LITS_Tumor_Dataset(BaseSegDataset):
    """ChestXray dataset.

    In segmentation map annotation for ChestXray, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    # METAINFO = dict(
    #     classes=('background', 'liver', 'abnormal'),
    #     palette=[[120, 120, 120], [6, 230, 230],  [255, 255, 255]])
    METAINFO = dict(
        classes=('background', 'liver', 'tumor'),
        palette=[[120, 120, 120], [6, 230, 230], [255, 255, 255]])
    def __init__(self, **kwargs) -> None:
        super().__init__(
            img_suffix='.nii.gz',
            seg_map_suffix='.nii.gz',
            reduce_zero_label=False,
            **kwargs)
        assert self.file_client.exists(self.data_prefix['img_path'])