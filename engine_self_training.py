import math
import sys
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils.utils as utils
from timm.utils import accuracy 
from tqdm import tqdm
from utils.center import get_center, get_weights, refine_pseudoDA, get_center_v2
import gc
gc.collect()
torch.cuda.empty_cache()

CosineLoss = nn.CosineEmbeddingLoss()

def train_one_epoch(args, model: torch.nn.Module, 
                    data_loader_u: Iterable, optimizer: torch.optim.Optimizer,
                    amp_autocast, device: torch.device, epoch: int, 
                    loss_scaler, 
                    lr_schedule_values, 
                    train_config,
                    start_steps=None,
                    memory=None,
                    prob_list=None
                    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    # -----------------------------------------------------------------------------------------------
    for data_iter_step, (img, true_labels, idx) in enumerate(metric_logger.log_every(data_loader_u, print_freq, header)):
        # ----------------------- unlabelled --------------------------------------------------------
        img_weak = img[0].to(device, non_blocking=True)
        img_strong = img[1].to(device, non_blocking=True)
        true_labels = true_labels.to(device, non_blocking=True)
        # ----------------------- assign learning rate for each step ---------------------------------
        it = start_steps + data_iter_step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None: 
                    param_group["lr"] = lr_schedule_values[it]
        # --------------------------------------------------------------------------------------------
        with torch.no_grad():
            feat_weak = model(img_weak)
            logit_txt = 100. * feat_weak @ model.get_classifier().t()
            probs_txt = F.softmax(logit_txt,dim=-1)
        # -------------------------------Refine pseudo labels-------------------------------------------------------------
            prob_list.append(probs_txt.mean(0))
            if len(prob_list)>32:
                prob_list.pop(0)
            probs, _, probs_centre = refine_pseudoDA(train_config, probs_txt, prob_list, model.get_center().to(device), feat_weak)
            # probs = probs_txt
            pseudo_labels = torch.argmax(probs, -1)
        # -------------------------------get weights -------------------------------------------------------------
            weights_txt = get_weights(probs_txt, pseudo_labels)
            weights_centre = get_weights(probs_centre, pseudo_labels)
            weights = (weights_centre*weights_txt).to(device)
            # weights = (weights_txt).to(device)
        #---------------------------------------------------------------------------------------
        with amp_autocast(): 
            feat_strong = model(img_strong)
            logits_strong_txt = 100. * feat_strong @ model.get_classifier().t()
            loss = (weights*F.cross_entropy(logits_strong_txt, pseudo_labels, reduction='none')).mean()
            # loss = (F.cross_entropy(logits_strong_txt, pseudo_labels, reduction='none')).mean()
            #----------------------------------------------------------------------------------
            loss += -train_config['reg']*(torch.log((F.softmax(logits_strong_txt,dim=-1)).mean(0))).mean()
            #----------------------------------------------------------------------------------
            logits = train_config['w'] * model.get_center().to(device) @ model.get_classifier().t()
            labels = torch.arange(len(logits)).to(device)
            loss += train_config['align']*F.cross_entropy(logits, labels)
        # ------------------------------------------------------ -------------------------------------------------------------------------
        memory['features'][idx] = feat_weak.detach().cpu()
        memory['labels'][idx] = pseudo_labels.detach().cpu()
        #------------------------------------------------------ For the record -----------------------------------------------------------
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc_selected=(pseudo_labels == true_labels).float().mean().item()*100)
        metric_logger.update(acc_student=(F.softmax(logits_strong_txt, -1).argmax(-1) == true_labels).float().mean().item()*100)
        # ---------------------------------------------------------------------
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)
        ## ---------------------- for grad -------------------------------------
        optimizer.zero_grad()
        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=1.0, parameters=model.parameters(), create_graph=False)
            metric_logger.update(grad_norm=grad_norm)
        else:                   
            loss.backward(create_graph=False)       
            optimizer.step()
        torch.cuda.synchronize()
    print('-----------------------------------------------------------------------')
    print(f"Averaged stats: {epoch} : {metric_logger}")
    model.center_init_fixed(get_center(args, memory, model.get_classifier())) 
    print('-----------------------------------------------------------------------')
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, \
    memory, prob_list

@torch.no_grad()
def evaluate(data_loader, model, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        # compute output
        feat_test = model(images)
        output = 100. * feat_test @ model.get_classifier().t()
        acc = accuracy(output, target)[0]
        metric_logger.meters['acc'].update(acc.item(), n=images.shape[0]) 
    print(f"* Acc@1 {metric_logger.acc.global_avg:.3f}")   
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


