# 主U上加了隐藏层（forward简化了v2中的多次判断）
import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1,groups=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate,groups=groups,bias=False)
        self.bn_s1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.relu_s1 = nn.GELU(approximate='tanh')

    def forward(self,x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))
    

def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[-2:], mode='bilinear')

# 定义双卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            REBNCONV(in_channels, mid_channels),
            REBNCONV(mid_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

# 定义上采样模块
class Up(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch):
        super(Up, self).__init__()
        self.conv1 = REBNCONV(in_ch + mid_ch, mid_ch)
        self.conv2 = REBNCONV(mid_ch + out_ch, out_ch)

    def forward(self, deepx, skipx):
        x = F.interpolate(deepx, size=skipx.shape[-2:], mode='bilinear', align_corners=True)
        h = self.conv1(torch.cat((x, x), 1))  # 使用 x 替代 h_prev
        y = self.conv2(torch.cat((h, skipx), 1))
        return y

### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2, groups=2)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2, groups=2)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2, groups=2)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=2, groups=2)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1, groups=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1, groups=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1, groups=2)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2, groups=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4, groups=2)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=6, groups=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4, groups=2)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2, groups=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1, groups=2)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


class self_net(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, mid_ch=64):
        super(self_net, self).__init__()

        # 编码器部分
        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # 解码器部分
        self.up5 = Up(64, 64, mid_ch)
        self.stage5d = RSU4F(64, 16, 64)

        self.up4 = Up(64, 64, mid_ch)
        self.stage4d = RSU4(64, 16, 64)

        self.up3 = Up(64, 64, mid_ch)
        self.stage3d = RSU5(64, 16, 64)

        self.up2 = Up(64, 64, mid_ch)
        self.stage2d = RSU6(64, 16, 64)

        self.up1 = Up(64, 64, mid_ch)
        self.stage1d = RSU7(64, 16, 64)

        # 侧输出
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1, bias=False)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1, bias=False)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1, bias=False)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1, bias=False)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1, bias=False)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1, bias=False)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1, bias=False)

    def forward(self, x):
        hx = x

        # 编码器
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # 解码器
        hx5d = self.up5(hx6, hx5)
        hx5d = self.stage5d(hx5d)

        hx4d = self.up4(hx5d, hx4)
        hx4d = self.stage4d(hx4d)

        hx3d = self.up3(hx4d, hx3)
        hx3d = self.stage3d(hx3d)

        hx2d = self.up2(hx3d, hx2)
        hx2d = self.stage2d(hx2d)

        hx1d = self.up1(hx2d, hx1)
        hx1d = self.stage1d(hx1d)

        # 侧输出
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return d0