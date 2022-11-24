import torch
import torch.nn as nn
from mmseg.models.utils import resize
class Encoder_Decoder(nn.Module):
    def __init__(self,
                 backbone,
                 decode_head,
                 ori_size):
        super(Encoder_Decoder, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.ori_size = ori_size
    def forward(self, x):
        x = self.backbone.forward(x)
        seg_logits = self.decode_head.forward(x)
        seg_logits = resize(
            seg_logits,
            size=self.ori_size,
            mode='bilinear')
        # pred = seg_logits.argmax(dim=1, keepdim=True).squeeze(0).squeeze(0)
        return seg_logits