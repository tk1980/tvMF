import argparse
import os
import random
import re

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter

from utils.train import validate, train, adjust_learning_rate, save_checkpoint
import utils.mydatasets as mydatasets

import models
from models.WrapperNet import WrapperNet

import config.model_config as cf
import config.loss_config as lcf


parser = argparse.ArgumentParser(description='Training')
#Data
parser.add_argument('--data', metavar='DIR',default='./datasets/imagenetlt/', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET',default='inat2018', type=str,
                    choices=['imagenet', 'imagenetlt', 'inat2018', 'inat2019'], help='dataset name')

#Network
parser.add_argument('--net-config', default='ResNet50Feature', type=str, metavar='CONFIG',
                    help='config name in network config file (default: ResNet50Feature)')
parser.add_argument('--loss-config', default='tvMF_k16', type=str, metavar='CONFIG',
                    help='config name in loss config file (default: tvMF_k16)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

#Utility
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--out-dir', default='./results', type=str,
                    help='path to output directory (default: ./)')
parser.add_argument('--save-all-checkpoints', dest='save_all_checkpoints', action='store_true',
                    help='save all the checkpoints')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

#Mode
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main():
    # performance stats
    stats = {'train_err1': [], 'train_err5': [], 'train_loss': [],
            'test_err1': [],  'test_err5': [],  'test_loss': []}

    # parameters
    args = parser.parse_args()
    
    args.num_classes = {'imagenet':1000, 'imagenetlt':1000, 'inat2018':8142, 'inat2019':1010}[args.dataset]
    args.input_size = (1, 3, 224, 224)

    # parameters specified by config file
    dataset = re.sub('lt$|201[0-9]$', '', args.dataset) # configs are shared among some datasets of the same type
    params = cf.__dict__[args.net_config]
    params.update(lcf.__dict__[args.loss_config])
    for name in ('arch', 'batch_size', 'lrs', 'opt_params', 'loss_params'):
        if name not in params.keys():
            print('parameter \'{}\' is not specified in config file.'.format(name))
            return
        args.__dict__[name] = params[name]
        print(name+':', params[name])
    args.start_epoch = 0
    args.epochs = len(args.lrs)
    args.train_transform = cf.train_transform[dataset]
    args.test_transform = cf.test_transform[dataset]
    print('train_transform:', args.train_transform)
    print('test_transform:', args.test_transform)

    # output directory
    if not args.evaluate:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # create model
    print("=> creating model '{}'".format(args.arch))
    feat = models.__dict__[args.arch]()
    model = WrapperNet(model=feat, num_classes=args.num_classes, **args.loss_params)
    print(model)

    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'logs'))
    writer.add_graph(model, (torch.randn(args.input_size), torch.zeros(1, dtype=torch.int64)))
    writer.close()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # only model
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            model.feat.load_state_dict(checkpoint['feat_state_dict'])
            model.fc_loss.load_state_dict(checkpoint['fc_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) for model"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # DataParallel will divide and allocate batch_size to all available GPUs
    model.feat = torch.nn.DataParallel(model.feat)
    model.cuda()

    # Data loading code
    if args.dataset == 'imagenet': 
        # ImageNet
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val_dir')

        train_dataset = datasets.ImageFolder(
            traindir,
            args.train_transform
            )
        val_dataset = datasets.ImageFolder(
            valdir, 
            args.test_transform
            )
    elif args.dataset in ('imagenetlt', 'inat2018', 'inat2019'):
        train_dataset = mydatasets.ListDataset(args.data, './datasets/{}/train.txt'.format(args.dataset), transform=args.train_transform)
        val_dataset = mydatasets.ListDataset(args.data, './datasets/{}/val.txt'.format(args.dataset), transform=args.test_transform)


    # Data Sampling
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lrs[0], **args.opt_params)

    # optionally resume from a checkpoint
    if args.resume:
        # other state parameters
        if os.path.isfile(args.resume):
            args.start_epoch = checkpoint['epoch']
            stats = checkpoint['stats']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) for the others"
                  .format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    # Do Eval
    if args.evaluate:
        validate(val_loader, model, None, args, True)
        return

    # Do Train
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        trnerr1, trnerr5, trnloss = train(train_loader, model, None, optimizer, epoch, args)

        # evaluate on validation set
        valerr1, valerr5, valloss = validate(val_loader, model, None, args)

        # statistics
        stats['train_err1'].append(trnerr1)
        stats['train_err5'].append(trnerr5)
        stats['train_loss'].append(trnloss)
        stats['test_err1'].append(valerr1)
        stats['test_err5'].append(valerr5)
        stats['test_loss'].append(valloss)

        # remember best err@1
        is_best = valerr1 <= min(stats['test_err1'])

        # show and save results
        writer.add_scalar('LearningRate', lr, epoch)
        writer.add_scalar('Loss/train', trnloss, epoch)
        writer.add_scalar('Loss/test', valloss, epoch)
        writer.add_scalar('Error_1/train', trnerr1, epoch)
        writer.add_scalar('Error_1/test', valerr1, epoch)
        writer.add_scalar('Error_5/train', trnerr5, epoch)
        writer.add_scalar('Error_5/test', valerr5, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'feat_state_dict': model.feat.module.state_dict(),
            'fc_state_dict': model.fc_loss.state_dict(),
            'stats': stats,
            'optimizer' : optimizer.state_dict(),
        }, is_best, not args.save_all_checkpoints, filename=os.path.join(args.out_dir, 'checkpoint-epoch{:d}.pth.tar'.format(epoch+1)))


    # show the final results
    minind = stats['test_err1'].index(min(stats['test_err1']))
    print('\n *BEST* Err@1 {:.3f} Err@5 {:.3f}'.format(stats['test_err1'][minind], stats['test_err5'][minind]))
    writer.add_hparams({'dataset':args.dataset, 'arch':args.arch, 'bsize':args.batch_size}, 
                        {'best/err_1':stats['test_err1'][minind], 'best/err_5':stats['test_err5'][minind], 'best/epoch':minind})
    writer.close()


if __name__ == '__main__':
    main()