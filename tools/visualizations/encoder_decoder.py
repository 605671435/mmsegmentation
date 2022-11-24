import torch.nn as nn
from mmseg.models.utils import resize
class Encoder_Decoder(nn.Module):
    def __init__(self,
                 backbone,
                 decode_head,
                 seg_size):
        super(Encoder_Decoder, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.seg_size = seg_size
    def forward(self, x):
        x = self.backbone.forward(x)
        seg_logits = self.decode_head.forward(x)
        seg_pred = seg_logits.argmax(dim=0, keepdim=True)
        pred = resize(
            seg_pred,
            size=self.seg_size,
            mode='bilinear')
        return pred