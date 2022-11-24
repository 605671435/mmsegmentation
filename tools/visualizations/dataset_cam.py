# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
warnings.filterwarnings("ignore")
import os

import torch

import pkg_resources
import re

import mmcv
import numpy as np
from mmcv.transforms import Compose
from mmengine.config import Config
from torch.nn import BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm

from mmseg import digit_version
from mmseg.apis import init_model
from mmseg.utils import register_all_modules
from mmseg.models.utils import resize
from mmengine.visualization import WandbVisBackend
from mmengine.utils import ProgressBar
from mmseg.datasets.chestxray import ChestXrayDataset

try:
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
    from pytorch_grad_cam.activations_and_gradients import \
        ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}


def compute_metric(pred_sem_seg, gt_sem_seg, num_classes):
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

def build_reshape_transform(model):
    """Build reshape_transform for `cam.activations_and_grads`, which is
    necessary for ViT-like networks."""
    # ViT_based_Transformers have an additional clstoken in features
    def check_shape(tensor):
        if isinstance(tensor, tuple):
            assert len(tensor[0].size()) != 3, \
                (f"The input feature's shape is {tensor.size()}, and it seems "
                 'to have been flattened or from a vit-like network. '
                 "Please use `--vit-like` if it's from a vit-like network.")
            return tensor
        assert len(tensor.size()) != 3, \
            (f"The input feature's shape is {tensor.size()}, and it seems "
             'to have been flattened or from a vit-like network. '
             "Please use `--vit-like` if it's from a vit-like network.")
        return tensor

    return check_shape

def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once, In order to be compatible with mmcls,
    here we modify the ActivationsAndGradients object."""
    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers, use_cuda=use_cuda)
    # Release the original hooks in ActivationsAndGradients to use
    # ActivationsAndGradients.
    cam.activations_and_grads.release()
    cam.activations_and_grads = ActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)

    return cam


def get_layer(layer_str, model):
    """get model layer from given str."""
    cur_layer = model
    layer_names = layer_str.strip().split('.')

    def get_children_by_name(model, name):
        try:
            return getattr(model, name)
        except AttributeError as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    def get_children_by_eval(model, name):
        try:
            return eval(f'model{name}', {}, {'model': model})
        except (AttributeError, IndexError) as e:
            raise AttributeError(
                e.args[0] +
                '. Please use `--preview-model` to check keys at first.')

    for layer_name in layer_names:
        match_res = re.match('(?P<name>.+?)(?P<indices>(\\[.+\\])+)',
                             layer_name)
        if match_res:
            layer_name = match_res.groupdict()['name']
            indices = match_res.groupdict()['indices']
            cur_layer = get_children_by_name(cur_layer, layer_name)
            cur_layer = get_children_by_eval(cur_layer, indices)
        else:
            cur_layer = get_children_by_name(cur_layer, layer_name)

    return cur_layer


def show_cam_grad(grayscale_cam, src_img, title, out_path=None):
    """fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img) / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)

    if out_path:
        mmcv.imwrite(visualization_img, str(out_path))
        print('cam saved to {}'.format(str(out_path)))
    else:
        mmcv.imshow(visualization_img, win_name=title)


def get_default_traget_layers(model, args):
    """get default target layers from given model, here choose nrom type layer
    as default target layer."""
    norm_layers = []
    for m in model.backbone.modules():
        if isinstance(m, (BatchNorm2d, LayerNorm, GroupNorm, BatchNorm1d)):
            norm_layers.append(m)
    if len(norm_layers) == 0:
        raise ValueError(
            '`--target-layers` is empty. Please use `--preview-model`'
            ' to check keys at first and then specify `target-layers`.')
    # if the model is CNN model or Swin model, just use the last norm
    # layer as the target-layer, if the model is ViT model, the final
    # classification is done on the class token computed in the last
    # attention block, the output will not be affected by the 14x14
    # channels in the last layer. The gradient of the output with
    # respect to them, will be 0! here use the last 3rd norm layer.
    # means the first norm of the last decoder block.
    if args.vit_like:
        if args.num_extra_tokens:
            num_extra_tokens = args.num_extra_tokens
        elif hasattr(model.backbone, 'num_extra_tokens'):
            num_extra_tokens = model.backbone.num_extra_tokens
        else:
            raise AttributeError('Please set num_extra_tokens in backbone'
                                 " or using 'num-extra-tokens'")

        # if a vit-like backbone's num_extra_tokens bigger than 0, view it
        # as a VisionTransformer backbone, eg. DeiT, T2T-ViT.
        if num_extra_tokens >= 1:
            print('Automatically choose the last norm layer before the '
                  'final attention block as target_layer..')
            return [norm_layers[-3]]
    print('Automatically choose the last norm layer as target_layer.')
    target_layers = [norm_layers[-1]]
    return target_layers

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model, ori_size):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        self.ori_size = ori_size

    def forward(self, x):
        seg_logits = self.model(x)
        seg_logits = resize(
            seg_logits,
            size=self.ori_size,
            mode='bilinear')
        return seg_logits

def vis_cam(img_path, seg_path, config, checkpoint, target_layers, method):
    cfg = Config.fromfile(config)

    register_all_modules()
    # build the model from a config file and a checkpoint file
    model = init_model(cfg, checkpoint, device='cuda')

    # apply transform and perpare data
    transforms = Compose(cfg.val_dataloader.dataset.pipeline)
    data = transforms({'img_path': img_path,
                       'seg_map_path': seg_path,
                       'reduce_zero_label': None,
                       'seg_fields': []})
    src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)
    data['inputs'] = data['inputs'].unsqueeze(0)
    data = model.data_preprocessor(data, False)

    predict = model.predict(data['inputs'])

    gt_mask = data['data_samples'].gt_sem_seg.data
    pred_resized = resize(
        input=predict[0].pred_sem_seg.data.unsqueeze(0).float(),
        size=gt_mask.shape[-2:],
        mode='bilinear')
    dice, _ = compute_metric(pred_resized.squeeze(0), gt_mask, 2)
    mdice = np.mean(dice)

    pred_mask = np.float32(predict[0].pred_sem_seg.data.cpu().numpy())[0]

    # model = Encoder_Decoder(backbone=model.backbone, decode_head=model.decode_head, ori_size=data['inputs'].shape[-2:])
    model = SegmentationModelOutputWrapper(model, ori_size=data['inputs'].shape[-2:])

    # build target layers
    target_layers = [
        get_layer(layer, model) for layer in target_layers
    ]

    # activations_wrapper = ActivationsWrapper(model, target_layers)

    # init a cam grad calculator
    use_cuda = 'cuda'
    reshape_transform = build_reshape_transform(model)
    cam = init_cam(method, model, target_layers, use_cuda,
                   reshape_transform)

    # warp the target_category with ClassifierOutputTarget in grad_cam>=1.3.7,
    # to fix the bug in #654.
    targets = None

    grad_cam_v = pkg_resources.get_distribution('grad_cam').version
    if digit_version(grad_cam_v) >= digit_version('1.3.7'):
        from pytorch_grad_cam.utils.model_targets import \
            SemanticSegmentationTarget
        targets = [SemanticSegmentationTarget(1, pred_mask)]

    # calculate cam grads and show|save the visualization image
    grayscale_cam = cam(
        data['inputs'],
        targets,
        # eigen_smooth=eigen_smooth,
        # aug_smooth=aug_smooth
    )
    # return cam
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img) / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)

    return mdice, visualization_img
def main():
    target_layers = ['model']
    method = 'gradcam'
    attn_methods = ['se', 'cbam', 'psa', 'ex']
    val_img_path = './data/lung_segmentation/images/validation'
    val_ann_path = './data/lung_segmentation/annotations/validation'

    val_img_list = os.listdir(val_img_path)

    configs = ['./configs/psa/deeplabv3_r50-se-d8_4xb2-20k_chestxray-256x256.py',
               './configs/psa/deeplabv3_r50-cbam-d8_4xb2-20k_chestxray-256x256.py',
               './configs/psa/deeplabv3_r50-psa-d8_4xb2-20k_chestxray-256x256.py',
               './configs/psa/deeplabv3_r50-ex-d8_4xb2-20k_chestxray-256x256.py']

    checkpints = ['./work_dirs/deeplabv3_r50-se-d8_4xb2-20k_chestxray-256x256/best_mDice_iter_20000.pth',
                  './work_dirs/deeplabv3_r50-cbam-d8_4xb2-20k_chestxray-256x256/best_mDice_iter_18000.pth',
                  './work_dirs/deeplabv3_r50-psa-d8_4xb2-20k_chestxray-256x256/best_mDice_iter_18000.pth',
                  './work_dirs/deeplabv3_r50-ex-d8_4xb2-20k_chestxray-256x256/best_mDice_iter_20000.pth']

    vis_backend = WandbVisBackend(init_kwargs=dict(project='chest X-rays', name='vis_cam'),
                                  save_dir='./output')
    wandb = vis_backend.experiment
    columns = ["img", "se", "mdice_se", "cbam", "mdice_cbam", "psa", "mdice_psa", "ex", "mdice_ex", "gt"]
    print('INFO***create a wandb table***INFO')
    vis_table = wandb.Table(columns=columns)

    progress_bar = ProgressBar(len(val_img_list))

    classes = ChestXrayDataset.METAINFO['classes']

    num_classes = len(classes)
    class_id = list(range(num_classes - 1)) + [255]
    class_set = wandb.Classes([{'name': name, 'id': id}
                               for name, id in zip(classes, class_id)])

    for img in val_img_list:
        img_dir = os.path.join(val_img_path, img)
        seg_dir = os.path.join(val_ann_path, img)

        ori_img = mmcv.imread(img_dir)
        gt_img = mmcv.imread(seg_dir, flag='grayscale')

        annotated = wandb.Image(ori_img, classes=class_set,
                                masks={"ground_truth": {"mask_data": gt_img}})

        vis_imgs = []
        mdices = []
        for i, attn_method in enumerate(attn_methods):
            mdice, visualization_img = vis_cam(img_path=img_dir,
                                        seg_path=seg_dir,
                                        config=configs[i],
                                        checkpoint=checkpints[i],
                                        target_layers=target_layers,
                                        method=method)
            vis_imgs.append(visualization_img)
            mdices.append(mdice)
        vis_table.add_data(img,
                           wandb.Image(vis_imgs[0]),
                           mdices[0],
                           wandb.Image(vis_imgs[1]),
                           mdices[1],
                           wandb.Image(vis_imgs[2]),
                           mdices[2],
                           wandb.Image(vis_imgs[3]),
                           mdices[3],
                           annotated)
        progress_bar.update()
    wandb.log({"visualization_cam": vis_table})
    print('INFO***log table to wandb***INFO')
if __name__ == '__main__':
    main()
