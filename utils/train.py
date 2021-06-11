import os
import shutil
import time
from tqdm import tqdm

import torch


def train(train_loader, model, criterion, optimizer, epoch, args, verbose=True):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Err@1', ':6.2f')
    top5 = AverageMeter('Err@5', ':6.2f')

    if verbose:
        num_batches = len(train_loader)
        tqdm_batch = tqdm(total=num_batches, desc="[Train, Epoch {}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for images, target in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        if criterion is None:
            output, loss = model(images, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(100.0-acc1.item(), images.size(0))
        top5.update(100.0-acc5.item(), images.size(0))

        # compute gradient and do updating step
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            tqdm_batch.set_postfix({'Time': '{:.2f} ({:.2f})'.format(batch_time.val, batch_time.avg),
                                    'Data': '{:.2f} ({:.2f})'.format(data_time.val, data_time.avg),
                                    'Loss': '{:.2f} ({:.2f})'.format(losses.val, losses.avg), 
                                    'Err@1': '{:.2f} ({:.2f})'.format(top1.val, top1.avg), 
                                    'Err@5': '{:.2f} ({:.2f})'.format(top5.val, top5.avg)})
            tqdm_batch.update()

    if verbose:
        tqdm_batch.close()

    return top1.avg, top5.avg, losses.avg


def finetune(train_loader, model, criterion, optimizer, epoch, args, verbose=True):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Err@1', ':6.2f')
    top5 = AverageMeter('Err@5', ':6.2f')

    num_batches = len(train_loader.dataset) // train_loader.batch_size
    if verbose:
        tqdm_batch = tqdm(total=num_batches, desc="[Train, Epoch {}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # for images, target in train_loader:
    for idx, (images, target) in enumerate(train_loader):
        # Break when step equal to epoch step
        if idx == num_batches:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        if criterion is None:
            output, loss = model(images, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(100.0-acc1.item(), images.size(0))
        top5.update(100.0-acc5.item(), images.size(0))

        # compute gradient and do updating step
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose:
            tqdm_batch.set_postfix({'Time': '{:.2f} ({:.2f})'.format(batch_time.val, batch_time.avg),
                                    'Data': '{:.2f} ({:.2f})'.format(data_time.val, data_time.avg),
                                    'Loss': '{:.2f} ({:.2f})'.format(losses.val, losses.avg), 
                                    'Err@1': '{:.2f} ({:.2f})'.format(top1.val, top1.avg), 
                                    'Err@5': '{:.2f} ({:.2f})'.format(top5.val, top5.avg)})
            tqdm_batch.update()

    if verbose:
        tqdm_batch.close()

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, args, verbose=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Err@1', ':6.2f')
    top5 = AverageMeter('Err@5', ':6.2f')

    if verbose:
        num_batches = len(val_loader)
        tqdm_batch = tqdm(total=num_batches, desc="[Val]")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for images, target in val_loader:
            target = target.cuda(non_blocking=True)

            # compute output
            if criterion is None:
                output, loss = model(images, target)
            else:
                output = model(images)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(100.0-acc1.item(), images.size(0))
            top5.update(100.0-acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if verbose:
                tqdm_batch.set_postfix({'Time': '{:.2f} ({:.2f})'.format(batch_time.val, batch_time.avg),
                                        'Loss': '{:.2f} ({:.2f})'.format(losses.val, losses.avg), 
                                        'Top1': '{:.2f} ({:.2f})'.format(top1.val, top1.avg), 
                                        'Top5': '{:.2f} ({:.2f})'.format(top5.val, top5.avg)})
                tqdm_batch.update()

        if verbose:
            tqdm_batch.close()
            print('\n *val* Err@1 {top1.avg:.3f} Err@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, save_last, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename),'model_best.pth.tar'))
    if save_last:
        last_filename = filename.replace('epoch'+str(state['epoch']), 'epoch'+str(state['epoch']-1))
        if os.path.isfile(last_filename):
            os.remove(last_filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the one specified by a user"""
    lr = args.lrs[epoch]
    for param_group in optimizer.param_groups:
        if 'lr_scale' in param_group.keys():
            param_group['lr'] = lr * param_group['lr_scale']
        else:
            param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)