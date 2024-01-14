import numpy as np
import sys
import argparse
from PIL import Image as PILImage
import torch
from torchvision import transforms
import torchvision.transforms as trn
import torchvision
import torchvision.datasets as dset
import torch.nn.functional as F

from models.resnet_cpn_ova import *
from models.wideresnet_cpn_ova import *


# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    import utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector and Misclassification Detector (OVA)',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--dataset', '-m', type=str, default='cifar-10', choices=['cifar-10', 'cifar-100'], help='Dataset name.')
parser.add_argument('--data-dir', default='datasets/', help='directory of dataset for training and testing')
# Loading details
parser.add_argument('--load', '-l', type=str, default='./snapshots', help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', default='ova', type=str, choices=['ova', 'sigmoid'], help='score options: ova|sigmoid')

parser.add_argument('--model', default='resnet18', choices=['resnet18', 'wrn-28-10', 'wrn-40-2'], help='directory of model for saving checkpoint')
parser.add_argument('--model-path', default='.checkpoint/res18-', help='directory of model for saving checkpoint')

parser.add_argument('--temp', default=1.0, type=float, help='temperature of Euclidian distance')
parser.add_argument('--thres_type', default='multi', type=str, choices=['multi', 'one', 'const'])
parser.add_argument('--global_thres', type=float, default=100, help='constant threshold for binary classifiers')

args = parser.parse_args()
print(args)
    
mean=[0.0, 0.0, 0.0]
std=[1.0, 1.0, 1.0]
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
if args.dataset == 'cifar-10':
    train_data = dset.CIFAR10(args.data_dir, train=True, transform=test_transform)
    test_data = dset.CIFAR10(args.data_dir, train=False, transform=test_transform)
    num_classes = 10
    mix_thres = -0.95
elif args.dataset == 'cifar-100':
    train_data = dset.CIFAR100(args.data_dir, train=True, transform=test_transform)
    test_data = dset.CIFAR100(args.data_dir, train=False, transform=test_transform)
    num_classes = 100
    mix_thres = -0.65
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False, 
                                          num_workers=args.prefetch, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'resnet18':
    net = ResNet18_cpn(num_classes=num_classes, logit_temperature=args.temp, thres_type=args.thres_type, global_thres=args.global_thres).cuda()
elif args.model == 'wrn-28-10':
    net = WideResNet_CPN(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.0, logit_temperature=args.temp, thres_type=args.thres_type).cuda()
elif args.model == 'wrn-40-2':
    net = WideResNet_CPN(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0, logit_temperature=args.temp, thres_type=args.thres_type).cuda()

# /////////////// Detection Prelims ///////////////
ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []
    list_softmax = []
    list_correct = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data = data.cuda()  
            dist, _, _ = net(data)
            logits = -(dist - net.rejection_threshold) * net.temperature_scale
            if args.score == 'ova':
                smax = to_np(logits)
                zeros = torch.zeros(logits.shape[0], 1).cuda()
                logits_extra = torch.cat((logits, zeros), dim=1)
                ind_prob = to_np(F.softmax(logits_extra, dim=1)[:, :-1])
                p_max_in = np.max(ind_prob, axis=1)
                p_ood = to_np(1 / (1 + torch.sum(torch.exp(logits), dim=1) ))
                score_mix = np.minimum(1.0 - p_ood, p_max_in - mix_thres)
                _score.append(-score_mix)
                list_softmax.extend(score_mix)
            elif args.score == 'sigmoid':
                probs = logits.sigmoid()
                posterior_binary_prob = probs
                probs = probs.log().cuda()
                pred = logits.data.max(1, keepdim=True)[1]
                prob, _pred = posterior_binary_prob.max(1)
                prob = to_np(prob)
                output = logits
                smax = to_np(posterior_binary_prob)
                _score.append(-np.max(smax, axis=1))
                list_softmax.extend(prob)      

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                for j in range(len(preds)):
                    if preds[j] == target[j]:
                        cor = 1
                    else:
                        cor = 0
                    list_correct.append(cor)

                if args.use_xent:
                    _right_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(to_np((output.mean(1) - torch.logsumexp(output, dim=1)))[wrong_indices])
                elif args.score == 'ova':
                    _right_score.append(-score_mix[right_indices])
                    _wrong_score.append(-score_mix[wrong_indices])
                elif args.score == 'sigmoid':
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
            else:  # else if OOD
                for j in range(len(data)):
                    cor = 0
                    list_correct.append(cor)
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy(), list_softmax, list_correct
    else:
        return concat(_score)[:ood_num_examples].copy(), list_softmax, list_correct

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        out_score, list_softmax_OOD, list_correct_OOD = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.dataset)
    else:
        print_measures(auroc, aupr, fpr, args.dataset)
    return list_softmax_OOD, list_correct_OOD

# AURC, EAURC
def calc_aurc_eaurc(softmax, correct):
    correctness = np.array(correct)
    softmax_max = np.array(softmax)
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

# testing OOD detection and miclassification detection
checkpoint = torch.load(args.model_path)
net.load_state_dict(checkpoint)     
net.eval()

in_score, right_score, wrong_score, list_softmax_ID, list_correct_ID = get_ood_scores(test_loader, in_dist=True)
num_right = len(right_score)
num_wrong = len(wrong_score)
print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
test_accuracy = 100 * num_wrong / (num_wrong + num_right)

# /////////////// End Detection Prelims ///////////////
print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')
# /////////////// Error Detection ///////////////
print('\n\nError Detection')
show_performance(wrong_score, right_score, method_name=args.dataset)
ID_aurc, ID_eaurc = calc_aurc_eaurc(list_softmax_ID, list_correct_ID)
# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []
aurc_list, e_aurc_list = [], []
# /////////////// Textures ///////////////
ood_data = dset.ImageFolder(root=args.data_dir+"ood_data/dtd/images",
                            transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                                                transforms.ToTensor(), ]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                        num_workers=4, pin_memory=True)
print('\n\nTexture Detection')
_, _ = get_and_print_results(ood_loader)
# /////////////// SVHN /////////////// # cropped and no sampling of the test set
ood_data = svhn.SVHN(root=args.data_dir+"ood_data", split="test",
                    transform=transforms.Compose(
                        [#trn.Resize(32), 
                        transforms.ToTensor(),]), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                        num_workers=2, pin_memory=True)
print('\n\nSVHN Detection')
_, _ = get_and_print_results(ood_loader)
# /////////////// Places365 ///////////////
ood_data = dset.ImageFolder(root=args.data_dir+"ood_data/Places",
                            transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                                                transforms.ToTensor(),]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                        num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
_, _ = get_and_print_results(ood_loader)
# /////////////// LSUN-C ///////////////
ood_data = dset.ImageFolder(root=args.data_dir+"ood_data/LSUN",
                            transform=transforms.Compose([transforms.ToTensor(),]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                            num_workers=1, pin_memory=True)
print('\n\nLSUN_C Detection')
_, _ = get_and_print_results(ood_loader)
# /////////////// LSUN-R ///////////////
ood_data = dset.ImageFolder(root=args.data_dir+"ood_data/LSUN_resize",
                            transform=transforms.Compose([transforms.ToTensor(),]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                        num_workers=1, pin_memory=True)
print('\n\nLSUN_Resize Detection')
_, _ = get_and_print_results(ood_loader)
# /////////////// iSUN ///////////////
ood_data = dset.ImageFolder(root=args.data_dir+"ood_data/iSUN",
                            transform=transforms.Compose([transforms.ToTensor(),]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                        num_workers=1, pin_memory=True)
print('\n\niSUN Detection')
_, _ = get_and_print_results(ood_loader)
# /////////////// Mean Results ///////////////
print('\n\nMean Test Results!!!!!')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.dataset) 

