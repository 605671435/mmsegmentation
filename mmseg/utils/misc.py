# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union
import os
import urllib

import numpy as np
import torch
import torch.nn.functional as F

from .typing import SampleList
from mmengine.utils import scandir

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')

def auto_arrange_images(image_list: list, image_column: int = 2) -> np.ndarray:
    """Auto arrange image to image_column x N row.
    Args:
        image_list (list): cv2 image list.
        image_column (int): Arrange to N column. Default: 2.
    Return:
        (np.ndarray): image_column x N row merge image
    """
    img_count = len(image_list)
    if img_count <= image_column:
        # no need to arrange
        image_show = np.concatenate(image_list, axis=1)
    else:
        # arrange image according to image_column
        image_row = round(img_count / image_column)
        fill_img_list = [np.ones(image_list[0].shape, dtype=np.uint8) * 255
                         ] * (
                             image_row * image_column - img_count)
        image_list.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(image_row):
            start_col = image_column * i
            end_col = image_column * (i + 1)
            merge_col = np.hstack(image_list[start_col:end_col])
            merge_imgs_col.append(merge_col)

        # merge to one image
        image_show = np.vstack(merge_imgs_col)

    return image_show


def get_file_list(source_root: str) -> [list, dict]:
    """Get file list.
    Args:
        source_root (str): image or video source path
    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(source_root).split('?')[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f'Downloading source file to {file_save_path}')
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type

def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def stack_batch(inputs: List[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3 or inputs[0].ndim == 4, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    if inputs[0].ndim == 4:
        is3d = True
    else:
        is3d = False

    padded_inputs = []
    padded_samples = []
    if is3d:
        inputs_sizes = [(img.shape[-3], img.shape[-2], img.shape[-1]) for img in inputs]
    else:
        inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            if is3d:
                dim = max(size[-3] - tensor.shape[-3], 0)
                padding_size = (0, width, 0, height, 0, dim)
            # (padding_left, padding_right, padding_top, padding_bottom)
            else:
                padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            if is3d:
                dim = max(max_size[-3] - tensor.shape[-3], 0)
                padding_size = (0, width, 0, height, 0, dim)
            else:
                padding_size = (0, width, 0, height)
        else:
            if is3d:
                padding_size = [0, 0, 0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        # if is3d:
        #     pad_img = pad_img[0]
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            if is3d:
                gt_sem_seg_3d = data_sample.gt_sem_seg_3d.data
                del data_sample.gt_sem_seg_3d.data
                data_sample.gt_sem_seg_3d.data = F.pad(
                    gt_sem_seg_3d, padding_size, value=seg_pad_val)
                data_sample.set_metainfo({
                    'img_shape': tensor.shape[-3:],
                    'pad_shape': data_sample.gt_sem_seg_3d.shape,
                    'padding_size': padding_size})
                padded_samples.append(data_sample)
            else:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(
                    gt_sem_seg, padding_size, value=seg_pad_val)

                data_sample.set_metainfo({
                    'img_shape': tensor.shape[-2:],
                    'pad_shape': data_sample.gt_sem_seg.shape,
                    'padding_size': padding_size})
                padded_samples.append(data_sample)
        else:
            padded_samples = None

    return torch.stack(padded_inputs, dim=0), padded_samples
