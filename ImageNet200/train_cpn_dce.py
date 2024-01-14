import csv
import argparse
import os
import random
import shutil
import time
import warnings
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from ImageFolder import *
import torch.nn.functional as F
import sklearn.metrics
import numpy as np
from collections import OrderedDict


class Counter(dict):
    def __missing__(self, key):
        return None

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--num_class', default=200, type=int, metavar='n',
                    help='number of training classes')
parser.add_argument('-a',
                    '--arch',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=91,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=512,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--wd',
                    '--weight-decay',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p',
                    '--print-freq',
                    default=500,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed',
                    default=None,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--method', default='dce', type=str, help='dce training')
parser.add_argument('--temp', type=float, default=1.0, help='tempeture')
parser.add_argument('--pl_weight', type=float, default=0.01, metavar='LR', help='temperature parameter')
parser.add_argument('--pl_normalized', action='store_true',  help='whether applying normalization for calculating pl loss')
parser.add_argument('--train_dir', default='/data/datasets/imagenet/train', type=str, help='train_dir')
parser.add_argument('--test_dir', default='/data/datasets/imagenet/val', type=str, help='test_dir')
parser.add_argument('--model-dir', default='checkpoint/imagenet/', help='directory of model for saving checkpoint')

criterion_ova = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


# Custom ResNet50
class Custom_ResNet(nn.Module):
    def __init__(self, arch, classnum):
        super(Custom_ResNet, self).__init__()
        self.resnet = models.__dict__[arch](num_classes=classnum)
        self.centers = nn.Parameter(torch.rand(classnum, self.resnet.fc.in_features))
        self.resnet.fc = nn.Identity()  

    def disatance(self, features, centers):
        f_2 = features.pow(2).sum(dim=1, keepdim=True)
        c_2 = centers.pow(2).sum(dim=1, keepdim=True)
        dist = f_2 - 2*torch.matmul(features, centers.transpose(0,1)) + c_2.transpose(0,1)
        return dist

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        features = torch.flatten(x, 1)
        out = self.disatance(features, self.centers)
        return out, features, self.centers


def main():
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
    root = os.path.abspath(args.model_dir)
    if os.path.isdir(root) == False: os.makedirs(root)
    output_folder = root + '/' + args.arch + '_' + args.method + '_' + str(args.num_class) + '_' + str(args.temp) + '_' + str(args.pl_weight) + '_dce'
    args.output_folder = output_folder
    if os.path.isdir(output_folder) == False: os.makedirs(output_folder)
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    args.local_rank = local_rank

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    best_acc1 = .0

    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23454',
                            world_size=args.nprocs,
                            rank=local_rank)
    # create model
    classnum = args.num_class
    print("=> creating model '{}'".format(args.arch))
    model = Custom_ResNet(args.arch, classnum)
    # model = models.__dict__[args.arch](num_classes=classnum)

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    cudnn.benchmark = True
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    class_index = list(range(args.num_class))
    train_dataset = ImageFolder(args.train_dir, transform_train, index=class_index)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=8,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_dataset = ImageFolder(args.test_dir, transform_test, index=class_index)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=8,
                                             pin_memory=True,
                                             sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, local_rank, args)
        return

    base_lr = 0.1  # Initial learning rate
    lr_strat = [30, 60]  # Epochs where learning rate gets decreased
    lr_factor = 0.1  # Learning rate decrease factor
    custom_weight_decay = 1e-4  # Weight Decay
    custom_momentum = 0.9  # Momentum
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_strat, gamma=lr_factor)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)
        lr_scheduler.step()
        print("epoch: ", epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank, args)
        # evaluate on validation set
        acc1 = validate(epoch, val_loader, model, criterion, local_rank, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
                
        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'best_acc1': best_acc1,
                }, is_best, filename=args.output_folder+'/checkpoint.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()


    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        batch_size = len(images)
        one_hot = torch.zeros(target.size(0), args.num_class).cuda().scatter_(1, target.view(-1, 1), 1)
        dist, features, centers = model(images)
        temper = torch.min(torch.tensor(( (epoch + 1) / 30 * args.temp, args.temp)))
        logits = -dist  * temper
        if args.pl_normalized:
            normalized_prototype = F.normalize(model.module.centers)
            normalized_feature = F.normalize(features, p=2, dim=1)
            distances_squared = torch.sum((normalized_feature.unsqueeze(1) - normalized_prototype.unsqueeze(0)) ** 2, dim=2)
            pl_loss = args.pl_weight * ((distances_squared * one_hot).sum())/2/batch_size
        else:
            pl_loss = args.pl_weight * ((dist * one_hot).sum())/2/batch_size
        
        loss_cls = criterion(logits, target)
        loss = loss_cls + pl_loss
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        torch.distributed.barrier()

        reduced_loss = reduce_mean(loss, args.nprocs)
        reduced_acc1 = reduce_mean(acc1, args.nprocs)
        reduced_acc5 = reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), images.size(0))
        top1.update(reduced_acc1.item(), images.size(0))
        top5.update(reduced_acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(epoch, val_loader, model, criterion, local_rank, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    correct = []
    conf = []
    logits_list = []
    labels_list = []
    softmax = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)

            # compute output
            batch_size = len(images)
            one_hot = torch.zeros(target.size(0), args.num_class).cuda().scatter_(1, target.view(-1, 1), 1)
            dist, _, _ = model(images)
            temper = torch.min(torch.tensor(( (epoch + 1) / 30 * args.temp, args.temp)))
            output = -dist * temper          
            
            # output = model(images)
            if args.method == 'OpenMix':
                output = output[:, :args.num_class]

            logits_list.append(output.detach().cpu())
            labels_list.append(target.detach().cpu())
            soft = F.softmax(output, dim=1)
            softmax.append(soft.detach().cpu().numpy())
            prob, pred = soft.max(1)
            correct.append(pred.cpu().eq(target.detach().cpu()).numpy())
            conf.append(prob.detach().cpu().view(-1).numpy())

            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, args.nprocs)
            reduced_acc1 = reduce_mean(acc1, args.nprocs)
            reduced_acc5 = reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    labels_list = labels.tolist()

    labels_onehot = one_hot_encoding(labels_list)
    softmax = np.concatenate(softmax, axis=0)
    correct = np.concatenate(correct, axis=0)
    conf = np.concatenate(conf, axis=0)
    groud = np.ones_like(correct)
    ece = ece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()
    conf_wrong = np.mean(conf[groud ^ correct])
    conf_correct = np.mean(conf[correct])

    results = metric(correct, conf, softmax, labels_onehot)
    print(results)

    logs_dict = OrderedDict(Counter(
        {
            "epoch": {"value": epoch, "string": f"{epoch:03}"},
            "test_acc": {
                "value": top1,
                "string": f"{(top1)}",
            },
            "conf_correct": {
                "value": round(conf_correct, 4),
                "string": f"{(round(conf_correct, 4))}",
            },
            "conf_wrong": {
                "value": round(conf_wrong, 4),
                "string": f"{(round(conf_wrong, 4))}",
            },
            "ece": {
                "value": round(100. * ece, 2),
                "string": f"{(round(100. * ece, 2))}",
            },
            "nll": {
                "value": round(nll, 4),
                "string": f"{(round(nll, 4))}",
            },
            "Brier": {
                "value": results['brier'],
                "string": f"{(results['brier'])}",
            },
            "AURC": {
                "value": results['AURC'],
                "string": f"{(results['AURC'])}",
            },
            "E-AURC": {
                "value": results['E-AURC'],
                "string": f"{(results['E-AURC'])}",
            },
            "AUROC": {
                "value": results['AUROC'],
                "string": f"{(results['AUROC'])}",
            },
            "AUPR_Success": {
                "value": results['AU_PR_POS'],
                "string": f"{(results['AU_PR_POS'])}",
            },
            "AUPR_Error": {
                "value": results['AU_PR_NEG'],
                "string": f"{(results['AU_PR_NEG'])}",
            },
            "FPR-95%-TPR": {
                "value": results['FPR-95%-TPR'],
                "string": f"{(results['FPR-95%-TPR'])}",
            },

        }
    ))

    # Print metrics
    print_dict(logs_dict)
    csv_writter(path=args.output_folder, dic=OrderedDict(logs_dict), start=epoch)
    os.chdir('../..')

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def OE_mixup(x_in, x_out, alpha=10.0):
    if x_in.size()[0] != x_out.size()[0]:
        length = min(x_in.size()[0], x_out.size()[0])
        x_in = x_in[:length]
        x_out = x_out[:length]
    lam = np.random.beta(alpha, alpha)
    x_oe = lam * x_in + (1 - lam) * x_out
    return x_oe, lam


class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


ece_criterion = ECELoss().cuda()
nll_criterion = nn.CrossEntropyLoss().cuda()


def csv_writter(path, dic, start):
    os.chdir(path)
    # Write dic
    if start == 0:
        mode = 'w'
    else:
        mode = 'a'
    with open('logs.csv', mode) as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        if start == 0:
            writer.writerow(dic.keys())
        writer.writerow([elem["string"] for elem in dic.values()])


def print_dict(logs_dict):
    str_print = ""
    for metr_name in logs_dict:
        str_print += f"{metr_name}={logs_dict[metr_name]['string']},  "
    print(str_print)


def metric(y, x, softmax, labels_onehot):
    # y: labels: 1 for positive 0 for negative
    # x: confidence

    results = dict()
    length = min(softmax.shape[1], labels_onehot.shape[1])
    brier_score = np.mean(np.sum((softmax[:, :length] - labels_onehot[:, :length]) ** 2, axis=1))


    results['brier'] = brier_score
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, x)
    results['AUROC'] = sklearn.metrics.auc(fpr, tpr)

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y, x)
    results['AU_PR_POS'] = sklearn.metrics.auc(recall, precision)
    # print(1-y)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(1 - y, -x)
    results['AU_PR_NEG'] = sklearn.metrics.auc(recall, precision)

    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    results['FPR-95%-TPR'] = fpr[idx_tpr_95]

    aurc, eaurc = calc_aurc_eaurc(y, x)
    results['AURC'] = aurc
    results['E-AURC'] = eaurc


    for n in results: results[n] = round(100. * results[n], 2)
    return results


def calc_aurc_eaurc(correctness, softmax_max):
    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

# Calc coverage, risk
def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print("AURC {0:.2f}".format(aurc*1000))
    print("EAURC {0:.2f}".format(eaurc*1000))

    return aurc, eaurc

def one_hot_encoding(label):
    cls = set(label)
    class_dict = {c: np.identity(len(cls))[i, :] for i, c in enumerate(cls)}
    one_hot = np.array(list(map(class_dict.get, label)))

    return one_hot


if __name__ == '__main__':
    main()