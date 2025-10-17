from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torchvision.transforms import v2
from typing import Callable, Optional
from pathlib import Path
from PIL import Image
import transform as T
import torch
import math


class NEU_SEG(Dataset):
    def __init__(self, root: str, istrain: bool, trans: Optional[Callable] = None):
        istrain = 'training' if istrain else 'test'
        srcDir = Path(root) / 'images' / istrain
        tgtDir = Path(root) / 'annotations' / istrain
        assert srcDir.is_dir() and tgtDir.is_dir()
        self.srcPaths = sorted(srcDir.glob('*.jpg'))
        self.tgtPaths = sorted(tgtDir.glob('*.png'))
        assert len(self.srcPaths) == len(self.tgtPaths)
        self.trans = v2.Identity() if trans is None else trans
        self.trans_src = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.trans_tgt = v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)])

    def __getitem__(self, idx):
        src = Image.open(self.srcPaths[idx]).convert('L')
        tgt = Image.open(self.tgtPaths[idx]).convert('L')
        src = self.trans_src(src)
        tgt = self.trans_tgt(tgt)
        src, tgt = self.trans(src, tgt)
        return src, tgt
    
    def __len__(self):
        return len(self.srcPaths)
    

def collate_fn(batch):
    srcs, tgts = [], []
    for x in batch:
        srcs.append(x[0])
        tgts.append(x[1])
    srcs = torch.stack(srcs, dim=0)
    tgts = torch.stack(tgts, dim=0)
    return srcs, tgts


def get_dataloader(root: str, istrain: bool, batch_size: int, trans: Optional[Callable]):
    data = NEU_SEG(root, istrain, trans)
    sampler = RandomSampler(data) if istrain else SequentialSampler(data)
    bsampler = BatchSampler(sampler, batch_size, drop_last=True)
    return DataLoader(data, batch_sampler=bsampler, collate_fn=collate_fn)


def get_normparam(root: str, istrain: bool):
    trans = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    data = NEU_SEG(root, istrain, trans)
    mean, std = 0., 0.
    for idx in range(len(data)):
        src = data[idx][0]
        mean += src.mean().item() / len(data)
        std += (src.square()-src.mean().square()).mean().item() / len(data)
    return mean, math.sqrt(std)


if __name__ == '__main__':
    root = '/data/NEU_Seg-main'
    trans = T.Compose([
        T.RandomCrop((0.3, 1.), (0.5, 2.)),
        T.RandomRotate((-180, 180), p=1.),
        T.RandomFlip(),
        T.Resize((256, 256)),
        T.GaussianNoise(0.1),
    ])
    data = NEU_SEG(root, istrain=True, trans=trans)[0]
    print('src in dataset:', 'size=', data[0].size, 'mode=', data[0].mode)
    print('tgt in dataset:', 'size=', data[1].size, 'mode=', data[1].mode)
    dataloader = get_dataloader(root, istrain=True, batch_size=4, trans=trans)
    for x, y in dataloader:
        print('src shape in dataloader:', x.shape)
        print('tgt shape in dataloader:', y.shape)
        break
