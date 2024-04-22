import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
assert torch.__version__ >= "1.10.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):
    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=cfg.reproducible)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)
    # 创建可视化的训练过程
    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    # 获取训练数据
    train_loader = get_dataloader(
        cfg.rec,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )
    # 创建识别模型
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()
    # 定义CombinedMarginLoss损失函数
    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )
    # 选择优化器，默认 sgd
    if cfg.optimizer == "sgd":
        # 使用随机梯度优化器
        '''
        用于处理部分全连接层的操作.
        margin_loss：使用 CombinedMarginLoss 损失函数
        cfg.embedding_size：表示嵌入向量的维度大小
        cfg.num_classes：表示分类类别的数量
        cfg.sample_rate：表示用于计算候选权重的采样比率
        False：表示不使用权重剪枝
        '''
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        '''
        实例化一个sgd优化器
        params：需要优化的模型参数的列表。主干网络（backbone）的参数，部分全连接层（PartialFC）的参数。
        lr：学习率
        momentum：动量因子。
        weight_decay：权重衰减（L2正则化）系数。
        '''
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adam":
        # 使用adam优化器
        '''
        用于处理部分全连接层的操作.
        margin_loss：使用 CombinedMarginLoss 损失函数
        cfg.embedding_size：表示嵌入向量的维度大小
        cfg.num_classes：表示分类类别的数量
        cfg.sample_rate：表示用于计算候选权重的采样比率
        False：表示不使用权重剪枝
        '''
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        '''
        实例化一个adamw优化器
        params：需要优化的模型参数的列表。主干网络（backbone）的参数，部分全连接层（PartialFC）的参数。
        lr：学习率
        weight_decay：权重衰减（L2正则化）系数。
        '''
        opt = torch.optim.Adam(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)# ,betas=(0.928,0.999)
    else:
        raise
    # 计算总批次 = 每个GPU上批次大小 × 总GPU数量
    cfg.total_batch_size = cfg.batch_size * world_size
    # 计算预热步数 = 总样本数 / 总批次数 × 预热周期数
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    # 计算总步数 = 总样本数 / 总批次数 × 总周期数
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch
    '''
    实例化一个学习率调度器
    optimizer：优化器对象opt，用于调整学习率。
    warmup_iters：预热步数，即预热阶段的总步数。
    total_iters：总步数，即训练的总步数。
    根据预热步数和总步数来调整优化器的学习率。预热阶段会逐渐增加学习率，以帮助模型更好地收敛
    '''
    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)
    start_epoch = 0
    global_step = 0

    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint
    # 记录 cfg 里的每个参数的名称和值
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))
    '''
    回调函数，用于进行验证并记录验证结果
    val_targets: 验证目标的配置参数，即cfg.val_targets。
    rec_prefix: 保存验证结果的前缀，即cfg.rec。
    summary_writer: 摘要写入器，用于记录验证结果的摘要信息。
    wandb_logger: WandB日志记录器，用于将验证结果记录到WandB平台上。
    '''
    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec,
        summary_writer=summary_writer)
    '''
    回调函数，用于打印训练过程中的日志信息
    frequent: 打印日志的频率，即cfg.frequent。
    total_step: 总步数，即cfg.total_step。
    batch_size: 批量大小，即cfg.batch_size。
    start_step: 起始步数，即global_step的值。
    writer: 摘要写入器，用于记录训练过程的摘要信息。
    '''
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step=global_step,
        writer=summary_writer
    )
    # 用于计算和记录平均值
    loss_am = AverageMeter()
    # 混合精度训练
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            # 设置数据加载器的随机种子为当前轮次的值，以确保每个轮次的数据顺序不同
            train_loader.sampler.set_epoch(epoch)
        # 遍历训练数据加载器中的每个批次
        for batch_idx, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            # 获取图像特征向量
            local_embeddings = backbone(img)
            # 计算分类损失值
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            # 根据是否启用混合精度训练，可以选择使用不同的梯度更新方式来更新模型参数。
            if cfg.fp16:
                # 使用混合精度训练
                # 对损失值进行缩放，并进行反向传播
                amp.scale(loss).backward()
                # 判断是否进行梯度累积
                if global_step % cfg.gradient_acc == 0:
                    # 对优化器的梯度进行反缩放，还原为真实的梯度
                    amp.unscale_(opt)
                    # 对模型的梯度进行裁剪，以防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt) # 使用优化器进行梯度更新
                    amp.update() # 更新缩放因子
                    opt.zero_grad() # 清空优化器中的梯度
            else:
                # 反向传播
                loss.backward()
                # 判断是否进行梯度累积（gradient accumulation）
                if global_step % cfg.gradient_acc == 0:
                    # 对模型参数的梯度进行裁剪，以防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()  # 使用优化器进行梯度更新
                    opt.zero_grad()  # 清空优化器中的梯度
            # 更新学习率
            lr_scheduler.step()
            # 确保评估阶段不进行梯度计算
            with torch.no_grad():
                # 更新平均损失值
                loss_am.update(loss.item(), 1)
                # 记录训练过程的相关信息
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
        with torch.no_grad():
                # 进行测试
                callback_verification(epoch+1, backbone,args.config)
        # 保存
        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,  # 训练周期
                "global_step": global_step,  # 全局步骤
                "state_dict_backbone": backbone.module.state_dict(),  # 模型状态
                "state_dict_softmax_fc": module_partial_fc.state_dict(),  # 部分全连接层状态
                "state_optimizer": opt.state_dict(),  # 优化器状态
                "state_lr_scheduler": lr_scheduler.state_dict()  # 学习率状态
            }
            # 保存
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            # GPU编号是0
            # 保存 训练的模型
            path_module = os.path.join(cfg.output, f"model.pt")
            torch.save(backbone.module.state_dict(), path_module)
        if cfg.dali:
            # 重载数据加载状态
            train_loader.reset()

    if rank == 0:
        # GPU编号是0
        # 保存 训练的模型
        path_module = os.path.join(cfg.output, "model_last.pt")
        torch.save(backbone.module.state_dict(), path_module)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, help="py config file",default='configs/pyramid-ir100-1210-300')
    main(parser.parse_args())