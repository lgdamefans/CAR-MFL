import os
import random
import numpy as np
import torch

import argparse

from algorithms.ServerTrainers import ClassificationTrainer
from algorithms.FedAvg import FedAvg
from algorithms.FedAvgIn import FedAvgIn, FedAvgInRAG, FedAvgNoPublic

parser = argparse.ArgumentParser(description='Federated Learning')

# 设置随机种子 确保实验结果可复现
def set_seed(seed):
     # 设置 PyTorch 的 CPU 随机种子
    torch.manual_seed(seed)
    # 设置 PyTorch 的 GPU 随机种子
    torch.cuda.manual_seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
    # 设置 Python 的随机种子
    random.seed(seed)
    
    #To let the cuDNN use the same convolution every time
    # 设置 PyTorch 的 cuDNN 随机种子
    #     deterministic = True: 确保每次卷积运算使用相同的算法
    # benchmark = False: 禁用 cuDNN 的自动调优功能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#wandb 实验数据记录工具初始化 使用教程：https://zhuanlan.zhihu.com/p/493093033
def init_wandb(args):
    import wandb
    name = f"{str(args.name)}"

    wandb.init(
        project="qualitative",
        name = name,
        resume = None,
        config=args
    )

    return wandb

#参数配置函数
def args():
    #实验名称
    parser.add_argument('--name', type=str, default='Test', help='The name for different experimental runs.')
    #保存路径
    parser.add_argument('--exp_dir', type=str, default='./experiments/',
                        help='Locations to save different experimental runs.')
    #服务端配置文件
    parser.add_argument('--server_config_path', type=str, default='configs/server_configs.yaml',
                        help='Location for server configs')
    #客户端配置文件
    parser.add_argument('--client_config_path', type=str, default='configs/client_configs.yaml',
                        help='Location for client configs')
    #通信轮数
    parser.add_argument('--comm_rounds', type=int, default=30)
    #随机种子
    parser.add_argument('--seed', type=int, default=42)
    #算法类型
    parser.add_argument('--algorithm', type=str, default='standalone', choices=['standalone', 'full', 'fedavg', 'fedavgln', 'fedavgRAG', 'fedavgzeropublic'],
                        help='Choice of Federated Averages')
    #多模态客户端总数
    parser.add_argument('--num_clients', type=int, default=10,
                        help='total number of multimodal clients')
    #图像客户端总数
    parser.add_argument('--img_clients', type=int, default=10,
                        help='total number of image clients')
    #文字客户端总数
    parser.add_argument('--txt_clients', type=int, default=10,
                        help='total number of text clients')
    #噪声水平
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='noise_level')
    #是否保存客户端
    parser.add_argument('--save_clients', action="store_true", default=False)
    #是否优化
    parser.add_argument('--use_refinement', action="store_true", default=False)

    
args()
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    wandb = init_wandb(args)
    # 根据args中选择算法 实际训练
    if args.algorithm == 'standalone':
        trainer = ClassificationTrainer(args, args.server_config_path, wandb)
        trainer.run_standalone()
    elif args.algorithm == 'fedavg':
        engine = FedAvg(args, wandb)
        engine.run()
    elif args.algorithm == 'fedavgln':
        engine = FedAvgIn(args, wandb)
        engine.run()
    elif args.algorithm == 'fedavgRAG':
        engine = FedAvgInRAG(args, wandb)
        engine.run()
    elif args.algorithm == 'fedavgzeropublic':
        engine = FedAvgNoPublic(args, wandb)
        engine.run()
    else:
        raise ValueError(f"Not implemented {args.algorithm}")