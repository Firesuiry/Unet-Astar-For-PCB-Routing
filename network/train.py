import copy
import logging
import time

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from network.data_loader import MyDataset
import torch

from network.unet import ResNetUNet
from network.unet_ori import UNet as UNet_ori
from network.unet_pp import UNet_2Plus as UNet_pp
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.nn as nn
from medpy import metric


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=1):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * 0.5 + dice * 0.5

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def calc_loss2(pred, target, metrics, bce_weight=1):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    hd = hausdorff_distance(pred, target)
    iou = iou_loss(pred, target)

    loss = bce * 0.5 + dice * 0.5

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['hd'] += hd * target.size(0)
    metrics['iou'] += iou.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def hausdorff_distance(pred, target):
    # calc hausdorff distance
    return metric.hd(pred.cpu().numpy(), target.cpu().numpy())


def iou_loss(pred, target):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1
        IoU = IoU + IoU1

    return IoU / b


def calc_params(pred, target):
    dice = dice_loss(pred, target)
    hd = hausdorff_distance(pred, target)
    iou = iou_loss(pred, target)
    return dice, hd, iou


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, device, dataloaders, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            metrics = defaultdict(float)
            epoch_samples = 0
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = calc_loss(outputs, labels, metrics)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            # scheduler.step()
                    # statistics
                    epoch_samples += inputs.size(0)
            if phase == 'train':
                scheduler.step()
            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # save to disk
                torch.save(model.state_dict(), 'best_val_model.pth')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def eval_model(model, optimizer, device, dataloaders, model_type):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    since = time.time()
    # Each epoch has a training and validation phase

    phase = 'val'
    model.eval()  # Set model to evaluate mode
    metrics = defaultdict(float)
    epoch_samples = 0
    s_dice, s_hd, s_iou = 0, 0, 0
    with tqdm(dataloaders[phase], unit="batch") as tepoch:
        for inputs, labels in tepoch:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = calc_loss2(outputs, labels, metrics)
            # statistics
            epoch_samples += inputs.size(0)
    print(f'model_type:[{model_type}]')
    print_metrics(metrics, epoch_samples, phase)
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def eval(dataset_path, model_types, batch_size=16):
    logging.info('start training')
    dataset = MyDataset(dataset_path, check=False, max_num=1000)
    train_set, val_set = dataset.get_train_and_test_Dataset(0.1)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda'
    print(device)

    for model_type in model_types:
        if model_type == '':
            model = ResNetUNet(in_ch=3, out_ch=2).to(device)
        elif model_type == 'ori':
            model = UNet_ori(n_channels=3, n_classes=2).to(device)
        elif model_type == 'pp':
            model = UNet_pp(in_channels=3, n_classes=2).to(device)
        else:
            raise Exception('model type error')

        model_path = model_type + 'best_model.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        eval_model(model, None, device, dataloaders, model_type)


def train(dataset_path, model_type='', dataset_max_num=-1, epoch=60, batch_size=16):
    logging.info('start training')
    dataset = MyDataset(dataset_path, check=False, max_num=dataset_max_num)
    train_set, val_set = dataset.get_train_and_test_Dataset(0.1)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda'
    print(device)

    num_class = 2
    if model_type == '':
        model = ResNetUNet(in_ch=3, out_ch=2).to(device)
    elif model_type == 'ori':
        model = UNet_ori(n_channels=3, n_classes=2).to(device)
    elif model_type == 'pp':
        model = UNet_pp(in_channels=3, n_classes=2).to(device)
    else:
        raise Exception('model type error')

    # freeze backbone layers
    # for l in model.base_layers:
    #    for param in l.parameters():
    #        param.requires_grad = False

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.75)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=epoch, device=device, dataloaders=dataloaders)
    # save model
    torch.save(model.state_dict(), model_type + 'best_model.pth')


if __name__ == '__main__':
    train()
