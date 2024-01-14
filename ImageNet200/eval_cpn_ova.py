import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torchvision.models as models

from ImageFolder import *
from collections import OrderedDict

to_np = lambda x: x.data.cpu().numpy()
logsoftmax = torch.nn.LogSoftmax(dim=-1)
auroc_list, aupr_list, fpr_list = [], [], []
aurc_list, e_aurc_list = [], []

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

def iterate_data_msp(data_loader, model, in_dist=True):
    confs = []
    list_softmax = []
    list_correct = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for x, y in data_loader:
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)
            
            if args.score_OOD == 'ova':
                mix_thres = -0.00
                smax = to_np(logits)
                zeros = torch.zeros(logits.shape[0], 1).cuda()
                logits_extra = torch.cat((logits, zeros), dim=1)
                ind_prob = to_np(F.softmax(logits_extra, dim=1)[:, :-1])
                p_max_in = np.max(ind_prob, axis=1)
                p_ood = to_np(1 / (1 + torch.sum(torch.exp(logits), dim=1) ))
                score_mix = np.minimum(1.0 - p_ood, p_max_in - mix_thres)
                confs.extend(score_mix)
                list_softmax.extend(score_mix)               
            elif args.score_OOD == 'sigmoid':
                conf, _ = torch.max(logits.sigmoid(), dim=-1)
                confs.extend(conf.data.cpu().numpy())
                smax = to_np(logits.sigmoid())              
                list_softmax.extend(np.max(smax, axis=1))
            if in_dist:
                preds = np.argmax(to_np(logits), axis=1)
                targets = y.numpy().squeeze()
                for j in range(len(preds)):
                    if preds[j] == y[j]:
                        cor = 1
                    else:
                        cor = 0
                    list_correct.append(cor)               
            else:  # else if OOD
                for j in range(len(x)):
                    cor = 0
                    list_correct.append(cor)
             
    return np.array(confs), list_softmax, list_correct


def test_acc(in_loader, model):
    correct = 0
    test_n = 0
    for x, y in in_loader:
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)
            pred = logits.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            test_n += y.size(0)
    return correct /  test_n * 100.0

# Custom ResNet50
class Custom_ResNet(nn.Module):
    def __init__(self, arch, classnum, logit_temperature, thres_type, global_thres=150):
        super(Custom_ResNet, self).__init__()
        self.resnet = models.__dict__[arch](num_classes=classnum)
        self.centers = nn.Parameter(torch.rand(classnum, self.resnet.fc.in_features))
        self.resnet.fc = nn.Identity()
        self.temperature_scale = torch.tensor(logit_temperature)
        if thres_type == 'multi':
            self.rejection_threshold = nn.Parameter(torch.ones(1, classnum))
        elif thres_type == 'one':
            self.rejection_threshold = nn.Parameter(torch.tensor(1.00))
        elif thres_type == 'const':
            self.rejection_threshold = torch.tensor(global_thres)   

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
        logits = -(out - self.rejection_threshold) * self.temperature_scale
        return logits


def test_denoising_ood(resnet, bs=64, device=torch.device('cuda:0')):
    global score
    global model_key
    score = args.score_OOD
    
    logdir = 'checkpoint/log/'
    from tools.test_utils import get_measures
    from tools import log
    logger = log.setup_logger(args, logdir)      
    
    batch_imagenet = 512
    Imagenet_workers = 8
    in_datadir = args.ID_dataset_dir
    out_datadir_bath = args.OOD_dataset_dir
    # Data loading code
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    crop_size = 224
    val_size = 256
    transform_test = tv.transforms.Compose([
        tv.transforms.Resize(val_size),
        tv.transforms.CenterCrop(crop_size),
        tv.transforms.ToTensor(),
        normalize,
    ])
    class_index = list(range(args.num_class))
    ind_dataset = ImageFolder(in_datadir + 'val', transform_test, index=class_index)
    ind_loader = torch.utils.data.DataLoader(ind_dataset, batch_size=batch_imagenet, shuffle=False, num_workers=Imagenet_workers, pin_memory=True, drop_last=False)

    resnet.eval()
    print("Running test...")
    
    acc = test_acc(ind_loader, resnet)
    print("acc: ", acc)
    
    column_i = 0
    OOD_dataset = 'Textures'
    out_datadir = out_datadir_bath + 'dtd/images'
    out_set = tv.datasets.ImageFolder(out_datadir, transform_test)
    out_loader = torch.utils.data.DataLoader(out_set, batch_size=batch_imagenet, shuffle=False,num_workers=Imagenet_workers, pin_memory=True, drop_last=False)
    
    logger.info("Processing in-distribution data...")
    in_scores, list_softmax_ID, list_correct_ID = iterate_data_msp(ind_loader, resnet, in_dist=True)
    calc_aurc_eaurc(list_softmax_ID, list_correct_ID)
    
    logger.info("Processing out-of-distribution data...")
    out_scores, list_softmax_OOD, list_correct_OOD = iterate_data_msp(out_loader, resnet, in_dist=False)      
    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))
    print("out_examples.shape: ", out_examples.shape)
    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    logger.info('============Results for {}-{}============'.format(score, OOD_dataset))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))
    column_i += 1
    logger.flush()    
    
    logger.info("Running test...")
    logger.flush()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--score_OOD', type=str, default='ova', choices=['ova', 'sigmoid'], help='score function for rejection')
    parser.add_argument('--model_path', type=str, default="checkpoint/cpn/resnet50_baseline_500_1.5_0.0_0.9/checkpoint.pth.tar", help='model path')
    parser.add_argument('--arch', '-a', default='resnet50')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--num_class', default=200, type=int, metavar='n', help='number of training classes')
    parser.add_argument('--ID_dataset_dir', default='/data/datasets/imagenet/', type=str, help='ID_test_dir')
    parser.add_argument('--OOD_dataset_dir', default='/data/home/wzliu/z_project/MOS/', type=str, help='OOD_test_dir')
    parser.add_argument('--thres_type', default='multi', type=str, choices=['multi', 'one', 'const'])
    parser.add_argument('--temp', type=float, default=1.0, help='tempeture')
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring and checkpointing.")

    args = parser.parse_args()
    
    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args


if __name__ == '__main__':
    args = parse_args_and_config()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    model = Custom_ResNet(args.arch, args.num_class, args.temp, args.thres_type).cuda()
    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path)    
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_name = k.replace('module.', '')
        new_state_dict[new_name] = v
    model.load_state_dict(new_state_dict)    
    test_denoising_ood(model)
