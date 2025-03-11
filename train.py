import argparse
import datetime
import numpy as np
import time
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import json
import os
from contextlib import suppress
import random
from pathlib import Path
from collections import OrderedDict
import utils.utils as utils
from utils.build_dataset import build_dataset
from utils.model import clip_classifier
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from utils.utils import plot_center
from engine_self_training import train_one_epoch, evaluate
from utils.center import build_memory
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser('MUST training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=1, type=int) 
    # CLIP parameters
    parser.add_argument("--template", default='templates.json', type=str)
    parser.add_argument("--classname", default='classes.json', type=str)
    parser.add_argument('--clip_model', default='ViT-B/32', help='pretrained clip model name') 
    parser.add_argument('--image_mean', default=(0.48145466, 0.4578275, 0.40821073)) 
    parser.add_argument('--image_std', default=(0.26862954, 0.26130258, 0.27577711)) 
    parser.add_argument('--input_size', default=224, type=int, help='images input size') 
    # training parameters
    parser.add_argument("--train_config", default='train_configs.json', type=str, help='training configurations') 
    # Optimizer parameters
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.65)
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    # Augmentation parameters  
    parser.add_argument('--train_crop_min', default=0.3, type=float)
    parser.add_argument('--color_jitter', type=float, default=0, metavar='PCT')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # Dataset parameters
    parser.add_argument('--nb_classes', default=0, type=int, help='number of the classification types')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
    parser.add_argument('--output_dir', default='', help='path to save checkpoint and log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    # distributed training parameters
    parser.add_argument('--amp', action='store_true')
    return parser.parse_args()

def main(args):
    #-------------------------------- Train config --------------------------------
    train_config_path = os.path.join("./json_files", args.train_config)
    with open(train_config_path, 'r') as train_config_file:
        train_config_data = json.load(train_config_file)
    train_config = train_config_data[args.dataset+'_'+args.clip_model]
    if not args.output_dir:
        args.output_dir = os.path.join('output',args.dataset)    
        args.output_dir = os.path.join(args.output_dir, 
                "NoProto%s_epoch%d_lr%.8f"%(args.clip_model[:5],train_config['epochs'], train_config['lr'])) 
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.output_dir:    
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(dict(args._get_kwargs())) + "\n")
    device = torch.device(args.device)
    # ----------------- fix the seed for reproducibility -----------------
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # ------------------- Train Dataset -------------------------------
    batch_size = train_config["model_patch_size"]
    dataset_train, len_original = build_dataset(is_train=True, args=args)
    print(f'Total number of training samples : { len(dataset_train) }')
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler     = sampler_train,
        batch_size  = batch_size,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
    len_data_loader_train = len(data_loader_train)
    args.len_original=len_original
    # -------------------------------- Eval Dataset --------------------------------
    dataset_val, _ = build_dataset(is_train=False, args=args)  
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler     = sampler_val,
        batch_size  = 4*batch_size,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    # -------------------------------- Build Model --------------------------------
    model = clip_classifier(args)
    dataset_name = args.dataset
    args.nb_classes = len(model.classnames)
    ## ------------------------ eman_model ----------------------------------------
    center, memory = build_memory(args, 
                                model, 
                                dataset_name, 
                                data_loader_train, 
                                len_original, 
                                model.model.embed_dim, 
                                )
    model.center_init_fixed(center)
    prob_list = []
    ## ------------------------  Freeze every thing except the layer norm ------------------------
    params = list()
    for name, param in model.named_parameters():
        param.requires_grad_(False)
        if "classifier" in name:
            param.requires_grad_(True) 
        if 'ln' in name or 'bn' in name:
            param.requires_grad = True
        if param.requires_grad:
            params.append((name, param))
            print(f'{name}')
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params
                    if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.1},
        {'params': [p for n, p in params
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # -------------------------------- optimizer --------------------------------
    args.lr           = train_config['lr'] * 1
    args.min_lr       = args.min_lr * 2
    args.epochs       = train_config['epochs']
    args.eval_freq    = train_config['eval_freq']
    model_without_ddp = model
    n_parameters      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('-----------------------------------------------------------------------')
    print(f'n_parameters : {n_parameters}')
    print('-----------------------------------------------------------------------')
    optimizer = optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    loss_scaler = None
    amp_autocast = suppress
    # -------------------------------- scheduler --------------------------------
    num_training_steps_per_epoch = len_data_loader_train
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    #--------------------------------- load Model --------------------------
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)
    # -------------------------------- Eval --------------------------------
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}%")
        exit(0)
    # -------------------------------- Train ----------------------------------------
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    ##------------------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        train_stats, memory, prob_list = train_one_epoch(
                                        args, model,
                                        data_loader_train, optimizer, amp_autocast, device, epoch, 
                                        loss_scaler = loss_scaler,  
                                        lr_schedule_values = lr_schedule_values,
                                        train_config=train_config,  
                                        start_steps=epoch * num_training_steps_per_epoch,
                                        memory = memory,
                                        prob_list= prob_list
                                    )    
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}%")
        if max_accuracy < test_stats["acc"]:
            max_accuracy = test_stats["acc"]
            if args.output_dir:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="best")
        print('-----------------------------------------------------------------------')
        print(f'Max accuracy: {max_accuracy:.2f}%')
        print('-----------------------------------------------------------------------')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
        #------------------------------------------------------------------------------------------
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

if __name__ == '__main__':
    opts = get_args()
    main(opts)