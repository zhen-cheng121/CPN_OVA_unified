import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from models.resnet_cpn_ova import *
from models.wideresnet_cpn_ova import *

import logging
import time
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training for convolutional prototype network (CPN)')
model_options = ['resnet18', 'wrn-28-10', 'wrn-40-2']
dataset_options = ['cifar-10', 'cifar-100']
# basic settings for training
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--lr_max', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--model', '-a', default='resnet18', choices=model_options)
parser.add_argument('--model-dir', default='checkpoint/test', help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=100, type=int, metavar='N', help='save frequency')
parser.add_argument('--dataset', '-d', default='cifar-10', choices=dataset_options)
parser.add_argument('--data-dir', default='datasets/', help='directory of dataset for training and testing')

# settings for CPN
parser.add_argument('--temp', type=float, default=2.0, metavar='LR', help='temperature parameter in CPN')
parser.add_argument('--temp_warm_epoch', type=int, default=40, metavar='LR', help='temperature parameter')
parser.add_argument('--thres_type', default='multi', type=str, choices=['multi', 'one', 'const'])
parser.add_argument('--global_thres', type=float, default=100, help='constant threshold for binary classifiers')
parser.add_argument('--pl_weight', type=float, default=0.05, metavar='LR', help='weight of prototype loss (PL)')
parser.add_argument('--pl_normalized', action='store_true',  help='whether applying normalization for calculating PL')
parser.add_argument('--alpha_ova', type=float, default=0.90, metavar='LR', help='weight of ova loss for the hybrid learning strategy')

args = parser.parse_args()
print(args)

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
if args.dataset == 'cifar-10':
    num_classes = 10
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
elif  args.dataset == 'cifar-100':
    num_classes = 100
    trainset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

criterion = nn.CrossEntropyLoss()
criterion_ova = nn.BCEWithLogitsLoss()

def eval_test(model, device, test_loader):
    model.eval()
    test_n = 0
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            batch_size = len(data)
            data, target = data.to(device), target.to(device)
            dist, _, _ = model(data)  # centers shape [10, 512], feature shape [64, 512], dist [64, 10]       
            logits = -(dist - model.rejection_threshold) * model.temperature_scale
            correct += (logits[:batch_size].max(1)[1] == target).sum().item()
            test_n += target.size(0)
    test_time = time.time()
    test_accuracy = correct
    return test_loss, test_accuracy, test_n, test_time


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr_max
    if epoch >= 100:
        lr = args.lr_max * 0.1
    if epoch >= 150:
        lr = args.lr_max * 0.01
    if epoch >= 200:
        lr = args.lr_max * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    if args.model == 'resnet18':
        model = ResNet18_cpn(num_classes=num_classes, thres_type=args.thres_type, 
                             logit_temperature=args.temp, global_thres=args.global_thres).to(device)
    elif args.model == 'wrn-28-10':
        model = WideResNet_CPN(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.0, 
                               thres_type=args.thres_type, logit_temperature=args.temp, global_thres=args.global_thres).to(device)
    elif args.model == 'wrn-40-2':    
        model = WideResNet_CPN(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0, 
                               thres_type=args.thres_type, logit_temperature=args.temp, global_thres=args.global_thres).to(device)        
    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    logger.info('Epoch \t Train Time \t Test Time \t LR \t Train Loss \t Train Reg \t Train Acc \t  Test Loss \t Test Acc')
    for epoch in range(1, args.epochs + 1):
        temp_lr = adjust_learning_rate(optimizer, epoch)
        model.temperature_scale = torch.min(torch.tensor((epoch / args.temp_warm_epoch * args.temp, args.temp)))   # temper warm-up
        model.train()
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_reg_loss = 0
        train_n = 0  
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = len(data)
            optimizer.zero_grad()
    
            # calculating OVA loss
            label = torch.cuda.LongTensor(target).view(batch_size, 1).to(device)
            one_hot = torch.zeros(batch_size, num_classes).to(device).scatter_(1, label, 1)
            dist, features, centers = model(data)  # centers shape [10, 512], feature shape [64, 512], dist [64, 10]       
            logits = -(dist - model.rejection_threshold) * model.temperature_scale
            ova_loss = criterion_ova(logits, one_hot) * num_classes
            
            # calculating PL
            if args.pl_normalized:
                normalized_prototype = F.normalize(model.centers)
                normalized_feature = F.normalize(features, p=2, dim=1)
                distances_squared = torch.sum((normalized_feature.unsqueeze(1) - normalized_prototype.unsqueeze(0)) ** 2, dim=2)
                pl_loss = args.pl_weight * ((distances_squared * one_hot).sum())/2/batch_size
            else:
                pl_loss = args.pl_weight * ((dist * one_hot).sum())/2/batch_size
            
            # calculating CE loss derived from Dempster–Shafer theory of evicence
            zeros = torch.zeros(logits.shape[0], 1).cuda()
            logits_ce = torch.cat((logits, zeros), dim=1)
            ce_loss = criterion(logits_ce, target)
            
            if args.model == 'resnet18' and args.dataset == 'cifar-10' and args.thres_type == 'multi':
                loss = (args.alpha_ova * ova_loss + (1 - args.alpha_ova) * ce_loss) / num_classes + pl_loss
            else:
                loss = args.alpha_ova * ova_loss + (1 - args.alpha_ova) * ce_loss + pl_loss    
            loss.backward()
            optimizer.step()
            train_loss += ova_loss.item() * target.size(0)
            train_reg_loss += pl_loss.item() * target.size(0)
            train_acc += (logits[:batch_size].max(1)[1] == target).sum().item()
            train_n += target.size(0)

        train_time = time.time()
        test_loss, test_accuracy, test_n, test_time = eval_test(model, device, test_loader)
        logger.info('%d \t \t \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, temp_lr,
                train_loss/train_n, train_reg_loss/train_n, train_acc/train_n,
                test_loss/test_n, test_accuracy/test_n)

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}.pth'))
            # torch.save(optimizer.state_dict(), os.path.join(model_dir, f'opt_{epoch}.pth'))


if __name__ == '__main__':
    main()
