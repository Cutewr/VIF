import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_train import train_one_epoch
from dataloader.dataloader_VIF import RGBTDataSet
from dataloader.dataloader_MEF import MEFDataSet
from dataloader.dataloader_MFF import MFFDataSet

import model.ViT_MAE as VIT_MAE_Origin

from util.ema import *
import yaml


def get_args_parser():
    # config path
    parser = argparse.ArgumentParser('TC-MoA', add_help=False)
    parser.add_argument('--config_path', default='config/base.yaml', type=str,
                        help='config_path to load')
    # Dataset parameters
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)
    # distributed training parameters
    parser.add_argument('--world_size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank',  type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True,
                        help='url used to set up distributed training')

    return parser

# Calculation of the number of frozen and unfrozen parameters to facilitate error checking
def count_parameters(model):
    freeze_layer = [[],0]
    unfreeze_layer = [[],0]
    MoA_layer =[[],0]
    windows_bais_layer = [[],0]
    fusion_layer = [[],0]
    unuesd_layer = [[],0]
    for name, parms in model.named_parameters():
        if parms.requires_grad==False:
            freeze_layer[0].append(name)
            freeze_layer[1] +=  parms.numel()
        else:
            unfreeze_layer[0].append(name)
            unfreeze_layer[1] +=  parms.numel()
            if "relative_position_bias_table" in name:
                windows_bais_layer[0].append(name)
                windows_bais_layer[1] +=  parms.numel()
            elif "MoA" in name:
                MoA_layer[0].append(name)
                MoA_layer[1] +=  parms.numel()
            elif "FusionLayer" in name:
                fusion_layer[0].append(name)
                fusion_layer[1] +=  parms.numel()
            else:
                unuesd_layer[0].append(name)
                unuesd_layer[1] +=  parms.numel()
    print("freeze_layer:", freeze_layer[0])
    print("unfreeze_layer:", unfreeze_layer[0])
    print("windows_bais_layer:", windows_bais_layer[0])
    print("MoA_layer:", MoA_layer[0])
    print("fusion_layer:", fusion_layer[0])
    print("unuesd_layer:", unuesd_layer[0])

    print("freeze_layer:", freeze_layer[1],"M:",freeze_layer[1]/1000000,"MB",freeze_layer[1]*4/1024/1024)
    print("unfreeze_layer:", unfreeze_layer[1],"M:",unfreeze_layer[1]/1000000,"MB",unfreeze_layer[1]*4/1024/1024)
    print("windows_bais_layer:", windows_bais_layer[1],"M:",windows_bais_layer[1]/1000000,"MB",windows_bais_layer[1]*4/1024/1024)
    print("MoA_layer:", MoA_layer[1],"M:",MoA_layer[1]/1000000,"MB",MoA_layer[1]*4/1024/1024)
    print("fusion_layer:", fusion_layer[1],"M:",fusion_layer[1]/1000000,"MB",fusion_layer[1]*4/1024/1024)
    print("unuesd_layer:", unuesd_layer[1],"M:",unuesd_layer[1]/1000000,"MB",unuesd_layer[1]*4/1024/1024)
            # print(name, parms.requires_grads)
   

def main(args,config):
    #Initialising distribution training parameters
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    print('config:',config)

    device = torch.device(config["device"])

    # fix the seed for reproducibility
    seed = config["seed"] + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # Initialize the dataset for each task
    if config["VIF"]:
        RGBT_dataset = RGBTDataSet(dataset_root_dir=None, \
                    dataset_dict=config['VIF_dataset_dict'], \
                    upsample = config['upsample'],arbitrary_input_size = config['arbitrary_input_size'])
    if config["MEF"]:
        MEF_dataset = MEFDataSet(dataset_root_dir=None,  \
                        dataset_dict=config['MEF_dataset_dict'])
    if config["MFF"]:
        MFF_dataset = MFFDataSet(dataset_root_dir=None,  \
                        dataset_dict=config['MFF_dataset_dict'])
    
   
    num_replicas = misc.get_world_size()
    global_rank = misc.get_rank()
    
    # Initialize the Sampler for each task
    if config["VIF"]:
        sampler_train_VIF = torch.utils.data.DistributedSampler(
            RGBT_dataset, num_replicas=num_replicas, rank=global_rank, shuffle=True
            )
    if config["MEF"]:
        sampler_train_MEF = torch.utils.data.DistributedSampler(
            MEF_dataset, num_replicas=num_replicas, rank=global_rank, shuffle=True
        )
    if config["MFF"]:
        sampler_train_MFF = torch.utils.data.DistributedSampler(
            MFF_dataset, num_replicas=num_replicas, rank=global_rank, shuffle=True
        )
        
    #Initialize log
    if global_rank == 0 and config["log_dir"] is not None:
        config["log_dir"] = os.path.join(config["log_dir"],config["method_name"])
        os.makedirs(config["log_dir"], exist_ok=True)
        log_writer = SummaryWriter(log_dir=config["log_dir"])
    else:
        #log_writer = None
        log_writer = SummaryWriter(log_dir=config["log_dir"])

    # Initialize the  dataloader for each task
    data_loader_list = {}
    if config["VIF"]:
        # 为VIF任务创建 数据加载器
        data_loader_train_VIF = torch.utils.data.DataLoader(
                RGBT_dataset, sampler=sampler_train_VIF,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                pin_memory=args.pin_mem,
                drop_last=False,
            )
        data_loader_list["VIF"] = data_loader_train_VIF
    if config["MEF"]:
        data_loader_train_MEF = torch.utils.data.DataLoader(
            MEF_dataset, sampler=sampler_train_MEF,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        data_loader_list["MEF"] =data_loader_train_MEF
    if config["MFF"]:
        data_loader_train_MFF = torch.utils.data.DataLoader(
            MFF_dataset, sampler=sampler_train_MFF,
            batch_size=int(config["batch_size"]),
            num_workers=config["num_workers"],
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        data_loader_list["MFF"] =data_loader_train_MFF

    
    # define the model
    models_mae =  VIT_MAE_Origin
    print("Model Type: VIT_MAE_Origin")
    # Initialize the model and load the parameters
    model = models_mae.__dict__[config["model_type"]](config)

    model.to(device)
    print("Loading weights_path.",config["pretrain_weight_path"])
    if config["load_start_epoch"]==0:
        freeze_list,unfreeze_list,ema_windows_list,ema_MoA_list,model = models_mae.load_pretrained_weights(model,model_name=None,weights_path=config["pretrain_weight_path"],\
                                  )
    else:
        freeze_list,unfreeze_list,ema_windows_list,ema_MoA_list,model = models_mae.load_pretrained_weights(model,model_ckp_path=config["ckp_path"],epoch=config["load_start_epoch"],
                                    model_name=None,weights_path=config["pretrain_weight_path"],\
                                 )

    print("---freeze_layer:--- ",freeze_list)
    print("---unfreeze_layer:--- ",unfreeze_list)
    count_parameters(model)
    model_without_ddp = model
    eff_batch_size = config["batch_size"]  * misc.get_world_size()
    print("Model = %s" % str(model_without_ddp))
    print("actual lr: %.2e" % config["lr"])
    print("effective batch size: %d" % eff_batch_size)
   
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set weightdecay as 0 for bias and norm layers
    # 为模型的不同参数进行权重衰减
    param_groups = optim_factory.add_weight_decay(model_without_ddp, config["weight_decay"])

    # param_groups：传入包含不同参数组的列表，每个参数组可能有不同的超参数（权重衰减率）
    # lr=config["lr"]：设置学习率，这个值从配置文件中读取（例如，config["lr"] = 1e-4）。
    # betas=(0.9, 0.95)：这是Adam优化器的动量参数，betas控制一阶矩估计和二阶矩估计的衰减率，通常设置为(0.9, 0.95)，这对大多数任务而言已经表现很好。
    # AdamW优化器结合了动量（betas）和自适应学习率（lr），并通过param_groups确保不同参数（如偏置和归一化层）有不同的优化策略（例如，偏置项不使用权重衰减）。
    optimizer = torch.optim.AdamW(param_groups, lr=config["lr"], betas=(0.9, 0.95))

    # print(optimizer) 混合精度训练【使用较低精度的浮点数（例如float16）来表示模型权重和梯度，从而加速训练过程并减少显存占用。】
    # NativeScaler负责缩放（scaling）损失值，以避免在混合精度训练时遇到数值不稳定（例如梯度溢出）的情况。
    loss_scaler = NativeScaler()

    # Stabilise training using EMA to prevent NaN
    # EMA（Exponential Moving Average） 是一种通过指数加权平均的方式来平滑模型的参数
    # 通常用于模型训练过程中，减少随机波动并提高最终模型的稳定性。
    if config["use_ema"]:
        state_dict = model.state_dict()
        
        ema_name_list = ["module.Alpha_encoder",
                    "module.Alpha_decoder",
                      ]
        ema_name_list += ["module."+i for i in ema_MoA_list]
        ema_name_list += ["module."+i for i in ema_windows_list]
        print("EMA_LIST:",ema_name_list)
        ema = EMA(0.99,ema_name_list)
        for name in ema_name_list:
            param_new = ema(name, state_dict[name])
    else:
        ema=None

    print("Start training for "+ str(config["epochs"])+ " epochs")
    start_time = time.time()
    for epoch in range(config["load_start_epoch"], config["epochs"]):
        if args.distributed:
            if config["VIF"]:
                data_loader_train_VIF.sampler.set_epoch(epoch)
            if config["MEF"]:
                data_loader_train_MEF.sampler.set_epoch(epoch)
            if config["MFF"]:
                data_loader_train_MFF.sampler.set_epoch(epoch)
        # 调用train_one_epoch训练每个epoch，使用当前的数据加载器、优化器、损失函数等训练模型
        # global_rank:zai
        train_stats = train_one_epoch(
            model, data_loader_list,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            config=config,global_rank=global_rank,ema=ema
        )
        #Save the model
        if config["output_dir"]:
            misc.save_model(
                config=config, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if config["output_dir"] and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # 用于定义和返回一个命令行参数解析器（Argument Parser）
    args = get_args_parser()
    # 解析命令行的参数，把参数结果放在args中
    args = args.parse_args()
    # 读取命令行中的config_path参数，打开config文件
    with open(args.config_path, 'r') as stream:
        # yaml.safe_load(stream)：这是 PyYAML 库提供的函数，用于从 stream（即配置文件）中加载 YAML 格式的配置。
        # yaml.safe_load() 会将 YAML 文件解析成 Python 对象（如字典、列表等）。
        # 这意味着配置文件内容会被加载到 config 变量中，可以在后续的代码中使用这些配置。
        config = yaml.safe_load(stream)
    main(args,config)
