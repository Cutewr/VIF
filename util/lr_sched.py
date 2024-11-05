# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

# 接收 优化器对象、epoch、配置
def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # 在预热（warm-up）阶段线性增加学习率，在预热后通过 余弦退火（Cosine Annealing） 策略逐渐降低学习率。
    if epoch < config["warmup_epochs"]:
        # 预热：线性增长到 config["lr"]
        lr = config["lr"] * epoch / config["warmup_epochs"]
    else:
        # 退火：生成-1到1的余弦曲线，+1 *0.5 使其映射到(0,1)区间。控制学习率从最大平滑地衰减到最小学习率
        lr = config["min_lr"] + (config["lr"] - config["min_lr"]) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config["warmup_epochs"]) / (config["epochs"] - config["warmup_epochs"])))
    for param_group in optimizer.param_groups:
        # 如果优化器中配置了 学习率缩放因子
        if "lr_scale" in param_group:
            # 乘优化器的 学习率缩放因子lr_scale
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
