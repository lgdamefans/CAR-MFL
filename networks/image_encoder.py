# 导入必要的库
import sys

import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def l2_normalize(tensor, axis=-1):
    """对张量进行L2归一化
    Args:
        tensor: 输入张量
        axis: 归一化的轴,默认为最后一维
    Returns:
        归一化后的张量
    """
    return F.normalize(tensor, p=2, dim=axis)

class EncoderResNet(nn.Module):
    """基于ResNet的编码器网络
    用于将图像编码为固定维度的嵌入向量
    """
    def __init__(self, embed_dim, cnn_type):
        """初始化编码器
        Args:
            embed_dim: 嵌入向量的维度
            cnn_type: 使用的ResNet类型,如'resnet50'
        """
        super(EncoderResNet, self).__init__()

        # 加载预训练的ResNet模型作为骨干网络
        self.cnn = getattr(models, cnn_type)(pretrained=True)

        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        # 保存原始的平均池化层
        self.avgpool = self.cnn.avgpool
        # 清空原始位置的池化层
        self.cnn.avgpool = nn.Sequential()

        # 添加全连接层,将特征映射到指定维度
        self.fc = nn.Linear(cnn_dim, embed_dim)

        # 移除原始ResNet的全连接层
        self.cnn.fc = nn.Sequential()
        # 设置所有参数可训练
        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = True

    def init_weights(self):
        """初始化全连接层的权重"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        """前向传播
        Args:
            images: 输入图像张量
        Returns:
            包含嵌入向量的字典
        """
        # 提取7x7的特征图
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        # 进行平均池化
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        # 通过全连接层
        out = self.fc(pooled)
        output = {}
        # L2归一化
        out = l2_normalize(out)
        output['embedding'] = out
        return output