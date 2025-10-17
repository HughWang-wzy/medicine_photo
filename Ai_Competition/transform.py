from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2
from typing import Tuple, Sequence, Optional
import random
import torch


def _rt2sp(x):
    if isinstance(x, tuple):
        return tuple(_rt2sp(_) for _ in x)
    elif 0. < x and x < 1.: return x-1
    elif 1. <= x: return 1-1/x


def _sp2rt(x):
    if isinstance(x, tuple):
        return tuple(_sp2rt(_) for _ in x)
    elif -1. < x and x < 0.: return x+1
    elif 0.<= x and x < 1.: return 1/(1-x)


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size
        self.mode = InterpolationMode.NEAREST
        self.antialias = False

    def __call__(self, src: torch.Tensor, tgt: torch.Tensor):
        src = F.resize(src, self.size, self.mode, antialias=self.antialias)
        tgt = F.resize(tgt, self.size, self.mode, antialias=self.antialias)
        return src, tgt


class RandomCrop:
    def __init__(self,
                 size_ratios: Tuple[float, float],
                 aspect_ratios: Tuple[float, float]):
        self.size_ratios = size_ratios
        self.aspect_ratios = aspect_ratios
        self.get_params = v2.RandomCrop.get_params

    def __call__(self, src: torch.Tensor, tgt: torch.Tensor):
        assert src.shape[-2:] == tgt.shape[-2:]
        aspect_ratios = _rt2sp(self.aspect_ratios)
        asp = _sp2rt(random.uniform(*aspect_ratios))
        a = min(1, max(self.size_ratios[0], self.size_ratios[0]*asp))
        b = max(a, min(self.size_ratios[1], self.size_ratios[1]*asp))
        w_ratio = random.uniform(a, b)
        h, w = int(src.shape[-2] * w_ratio / asp), int(src.shape[-1] * w_ratio)
        params = self.get_params(src, (h, w))
        src, tgt = F.crop(src, *params), F.crop(tgt, *params)
        return src, tgt


class RandomRotate:
    def __init__(self, angle_ranges: Tuple[int, int], p: float):
        self.angle_ranges = angle_ranges
        self.mode = InterpolationMode.NEAREST
        self.p = p

    def __call__(self, src, tgt):
        p = random.uniform(0., 1.)
        if p > self.p: return src, tgt
        ang = random.randint(*self.angle_ranges)
        src = F.rotate(src, ang, self.mode, True)
        tgt = F.rotate(tgt, ang, self.mode, True)
        return src, tgt

    
class RandomSquareRotate:
    def __init__(self, p: float = 1.):
        self.p = p
        self.mode = InterpolationMode.NEAREST

    def __call__(self, src, tgt):
        p = random.uniform(0., 1.)
        if p > self.p: return src, tgt
        ang = random.choice((-90, 0, 90, 180))
        src = F.rotate(src, ang, self.mode, True)
        tgt = F.rotate(tgt, ang, self.mode, True)
        return src, tgt


class RandomFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, src, tgt):
        p = random.uniform(0., 1.)
        if p < self.p:
            src = F.vertical_flip(src)
            tgt = F.vertical_flip(tgt)
        p = random.uniform(0., 1.)
        if p < self.p:
            src = F.horizontal_flip(src)
            tgt = F.horizontal_flip(tgt)
        return src, tgt


class GaussianNoise:
    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, src, tgt):
        src = src + torch.randn_like(src, device=src.device) * self.std
        return src, tgt
    

class Normalize:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def __call__(self, src, tgt):
        return F.normalize(src, self.mean, self.std), tgt
    

class Disturb:
    def __init__(self, colorjit: Optional[Tuple[float, float, float]],
                 gaussblur: Optional[Tuple[Sequence[int], Sequence[float]]],
                 sharpness: Optional[Tuple[Tuple[float, float], float]]):
        self.jit = colorjit
        self.gaussblur = gaussblur
        self.sharpness = sharpness
    
    def __call__(self, src, tgt):
        if self.jit is not None:
            p = random.uniform(1-self.jit[0], 1+self.jit[0])
            src = F.adjust_brightness(src, p)
            p = random.uniform(1-self.jit[1], 1+self.jit[1])
            src = F.adjust_contrast(src, p)
            p = random.uniform(1-self.jit[2], 1+self.jit[2])
            src = F.adjust_saturation(src, p)
        if self.gaussblur is not None:
            src = F.gaussian_blur(src, self.gaussblur[0], self.gaussblur[1])
        if self.sharpness is not None:
            p = random.uniform(0., 1.)
            if p < self.sharpness[1]:
                p = random.uniform(*self.sharpness[0])
                src = F.adjust_sharpness(src, p)
        return src, tgt
    

class Compose:
    def __init__(self, mods: list):
        self.mods = mods

    def __call__(self, src, tgt):
        for m in self.mods:
            src, tgt = m(src, tgt)
        return src, tgt


if __name__ == '__main__':
    trans = Resize((256, 256))
    import torch

    src = torch.rand((3, 128, 128))
    tgt1, tgt2 = trans(src, src)
    assert torch.all(tgt1 == tgt2)
    print(tgt1.shape, tgt2.shape)

    trans = RandomCrop((0.3, 1.), (0.5, 2.))
    tgt1, tgt2 = trans(src, src)
    assert torch.all(tgt1 == tgt2)
    print(tgt1.shape, tgt2.shape)

    trans = RandomRotate((-90, 90), 1.)
    tgt1, tgt2 = trans(src, src)
    assert torch.all(tgt1 == tgt2)
    print(tgt1.shape, tgt2.shape)

    trans = RandomFlip(0.5)
    tgt1, tgt2 = trans(src, src)
    assert torch.all(tgt1 == tgt2)
    print(tgt1.shape, tgt2.shape)

    trans = GaussianNoise(0.1)
    tgt1, tgt2 = trans(src, src)
    # assert torch.all(tgt1 == tgt2)
    print(tgt1.shape, tgt2.shape)
