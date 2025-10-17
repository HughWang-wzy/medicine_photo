import torch
from torch import nn
from torch.nn import functional as F


def upsample_like(src: torch.Tensor, tgt: torch.Tensor):
    return F.interpolate(src, tgt.shape[-2:], mode='bilinear', align_corners=True)


class ResNormConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilate=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=dilate, dilation=dilate, groups=groups)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.GELU(approximate='tanh')
        self.short = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        res = self.short(x)
        x = self.act(self.norm(self.conv(x)))
        return res + x
    

if __name__ == '__main__':
    from torchsummary import summary
    model = ResNormConv(3, 4).cuda()
    summary(model, (3, 256, 256))