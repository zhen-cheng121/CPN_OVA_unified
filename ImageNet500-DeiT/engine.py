# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.temp_warmup_S1:
        model.module.temperature_scale = (0.01 + (args.cpn_temp - 0.01) /2 * (1 - torch.cos(torch.tensor(epoch) * (3.141592653589793 / 15))))
        print("CPN temp: ", model.module.temperature_scale)
    elif args.temp_warmup_S2:
        model.module.temperature_scale = torch.min(torch.tensor(( (epoch + 1) / 15 * args.cpn_temp, args.cpn_temp)))
        print("CPN temp: ", model.module.temperature_scale)

    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        orginal_target = targets        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.mixup > 0 or args.cutmix > 0.:    
            if args.bce_loss:
                targets = targets.gt(args.cpn_target_threshold).type(targets.dtype)
        elif args.bce_loss:
            targets = torch.zeros(targets.size(0), args.nb_classes).cuda().scatter_(1, targets.view(-1, 1), 1)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                if args.cpn_Mclass and not args.ours_ova:
                    loss = criterion(samples, outputs, targets) * args.nb_classes
                elif args.ours_ova:
                    probs = outputs.sigmoid()
                    probs_others = (1 - probs).log().to(device)
                    probs = probs.log().to(device, non_blocking=True)
                    eye = (1 - torch.eye(probs.shape[1])).to(device)
                    probs =  torch.matmul(probs, torch.eye(probs.shape[1]).cuda())  + torch.matmul(probs_others,eye)
                    loss = -1*((probs * targets).sum())/args.batch_size
                    print("ova loss: ", loss)
                    # print("ova pytorch loss: ", loss)                    
                else:
                    loss = criterion(samples, outputs, targets)
                # loss = criterion(samples, outputs, targets)
            
                
                # loss = criterion(samples, outputs, targets)
                dist_proto = - outputs / model.module.temperature_scale + model.module.rejection_threshold
                pl_loss = args.cpn_pl * ((dist_proto * targets).sum())/ 2 / outputs.shape[0]
                print("PL: ", pl_loss)

                zeros = torch.zeros(outputs.shape[0], 1).to(device, non_blocking=True)
                logits = torch.cat((outputs, zeros), dim=1)
                if args.mixup > 0 or args.cutmix > 0.:
                    criterion_reg = SoftTargetCrossEntropy()
                    orginal_target_extend = torch.cat((orginal_target, zeros), dim=1)
                    loss_reg = criterion_reg(logits, orginal_target_extend)
                else:
                    criterion_reg = torch.nn.CrossEntropyLoss()
                    loss_reg = criterion_reg(logits, orginal_target)
                print("regCE: ", args.regCE * loss_reg)
                loss = args.ova_loss * loss + args.regCE * loss_reg + pl_loss
                # loss = args.regCE * loss_reg
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())
                

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        max_norm_value = float('inf')
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_value)
        grad_norm_centers = torch.nn.utils.clip_grad_norm_(model.module.centers, max_norm=max_norm_value)
        grad_norm_thres = torch.nn.utils.clip_grad_norm_(model.module.rejection_threshold, max_norm=max_norm_value)
        print("Gradient Norm:", grad_norm)
        print("Gradient Norm Centers:", grad_norm_centers)
        print("Gradient Norm Thres:", grad_norm_thres)
        # max_values, max_indices = torch.max(outputs, dim=1)
        # print("max_values: ", max_values)


        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        
        acc1_train, _ = accuracy(outputs, orginal_target, topk=(1,5))
        batch_size = outputs.shape[0]
        metric_logger.meters['acc1'].update(acc1_train.item(), n=batch_size)        
        
        # pl_value = pl_loss.item()
        # metric_logger.update(loss=pl_value)
        
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # test whether gradient exploding
        # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # print("Total Gradient Norm:", total_norm.item())
        param = model.module.centers[1,:]
        # gradient_class = param.grad
        # print("Value of Class 0:", param)

        # print("Parameter Value:", param)
        print("Parameter Norm:", torch.norm(param.data))
        print("Loss: ", loss_value)
        
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args = None):
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()    
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # target = target.float()
        # one_hot = target.gt(0.0).type(target.dtype)
        orginal_target = target
        one_hot = torch.zeros(orginal_target.size(0), args.nb_classes).cuda().scatter_(1, orginal_target.view(-1, 1), 1)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # loss = criterion(output, target)
            loss = criterion(output, one_hot) * args.nb_classes
            # loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction='mean')


        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
