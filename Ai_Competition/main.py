import random
import torch
import numpy

from data import get_dataloader, get_normparam
from engine import train_one_epoch, test_one_epoch, CELoss
import transform as T
from u2net import U2NETP

from logger import Logger


def fix_randomseed():
    random.seed(3407)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    numpy.random.seed(3407)


if __name__ == '__main__':
    # fix_randomseed()

    logger = Logger()
    root = '/home/ipad-ocr/NEU_Seg-main'
      
    mean_train, std_train = get_normparam(root, True)
    mean_test, std_test = get_normparam(root, False)

    model = U2NETP(1, 4).cuda()
    print('model params:', sum(p.numel() for p in model.parameters()))
    trans_train = T.Compose([
        T.Normalize(mean_train, std_train),
        T.RandomSquareRotate(),
        T.RandomFlip(),
        T.GaussianNoise(0.04),
    ])
    trans_test = T.Compose([
        T.Normalize(mean_test, std_test),
    ])
    train_dataloader = get_dataloader(root, istrain=True, batch_size=64, trans=trans_train)
    test_dataloader = get_dataloader(root, istrain=False, batch_size=8, trans=trans_test)
    critirion = CELoss(reduction='mean').cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-5)
    
    max_miou = 0.
    for epochIdx in range(180):
        epochIdx = epochIdx + 1
        print(epochIdx)
        trainloss = train_one_epoch(model, train_dataloader, critirion, optimizer)
        logger.update('train_loss', trainloss, epochIdx=epochIdx, stepIdx=None)
        
        if epochIdx % 2 == 0:
            testloss, testious = test_one_epoch(model, test_dataloader, critirion)
            logger.update('test_loss', testloss, epochIdx=epochIdx, stepIdx=None)
            logger.update('test_ious', testious, epochIdx=epochIdx, stepIdx=None)
        logger.save('./0926.yml')
