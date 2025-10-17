import torch
from torch import Tensor
from tqdm import tqdm


def cal_iu(preds: Tensor, tgts: Tensor, Class: int):
    preds = torch.argmax(preds, dim=1, keepdim=True)
    inter = ((preds == tgts) & (tgts == Class))
    union = ((preds == Class) | (tgts == Class))
    return torch.sum(inter).item(), torch.sum(union).item()


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    avgloss = 0
    for srcs, tgts in tqdm(dataloader, desc='Train', leave=False):
        srcs = srcs.cuda()
        tgts = tgts.cuda()
        preds = model(srcs)
        loss = criterion(preds, tgts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avgloss += loss.item() / len(dataloader)
    print('Train:', 'avgloss=', avgloss)
    return avgloss


@torch.no_grad()
def test_one_epoch(model, dataloader, criterion):
    model.eval()
    loss_history = {'totloss': 0, 'cnt': 0}
    ius_history = [
        {'name': 'Background', 'totInter': 0, 'totUnion': 0},
        {'name': 'Inclusions', 'totInter': 0, 'totUnion': 0},
        {'name': 'Patches', 'totInter': 0, 'totUnion': 0},
        {'name': 'Scratches', 'totInter': 0, 'totUnion': 0},
    ]


    for srcs, tgts in tqdm(dataloader, desc='Test', leave=False):
        srcs = srcs.cuda()
        tgts = tgts.cuda()
        preds = model(srcs).detach().clone()
        bloss = criterion(preds, tgts)
        loss_history['totloss'] += bloss.item()
        loss_history['cnt'] += 1
        
        for classid in range(len(ius_history)):
            i, u = cal_iu(preds, tgts, classid)
            ius_history[classid]['totInter'] += i
            ius_history[classid]['totUnion'] += u

    avgloss = loss_history['totloss'] / loss_history['cnt']
    avgious = [c['totInter'] / c['totUnion'] for c in ius_history]
    miou = sum(avgious[1:]) / len(avgious[1:])
    print('Val:', 'avgloss=', avgloss, 'avgious=', avgious, 'miou=', miou)
    avgious = [{'name': ius_history[classid]['name'], 'avgiou': avgious[classid]} for classid in range(len(ius_history))]
    avgious.append({'name': 'mIoU', 'avgiou': miou})
    return avgloss, avgious


class CELoss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.func = torch.nn.CrossEntropyLoss(*args, **kwargs)
    
    def forward(self, preds, tgts):
        num_classes = preds.shape[1]
        preds = preds.permute(0, 2, 3, 1).reshape(-1, num_classes)
        tgts = tgts.reshape(-1).long()
        return self.func(preds, tgts)
        # return torchvision.ops.sigmoid_focal_loss(preds, tgts)