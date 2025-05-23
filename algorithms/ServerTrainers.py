# 导入必要的库
import sys
import os
import pickle

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append("..")

from utils.config import parse_config
from datasets.mimic import MimicMultiModal
from datasets.iu_xray import IUXrayMultiModal
from networks import get_mmclf, get_tokenizer
from networks.optimizers import get_optimizer
from losses import get_criterion
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAUROC


class ClassificationTrainer:
    """分类模型训练器类"""
    def __init__(self, args, config_path, wandb=False):
        """
        初始化训练器
        args: 命令行参数
        config_path: 配置文件路径
        wandb: 是否使用wandb记录实验
        """
        self.args = args
        self.wandb = wandb

        # 加载配置文件
        self.config = parse_config(config_path)
        self.dset_name = self.config.dataset.dset_name
        self.load_data()
        self.load_model()

        # 初始化评估指标
        self.evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
        })
        self.cur_epoch = 0
        self.save_dir = os.path.join(self.args.exp_dir, "server")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.val_track = []

    def load_data(self):
        """加载数据集"""
        if self.dset_name == "mimic-cxr":
            # 加载MIMIC-CXR数据集
            partition_path = f'partitions/{self.dset_name}_{self.config.dataset.view}_{self.config.dataset.partition}.pkl'
            with open(partition_path, "rb") as f:
                data_partition = pickle.load(f)
            train_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "train")
            train_idx = data_partition["server"]
            self.train_set = Subset(train_set, train_idx)
            self.val_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "val")
            self.test_set = MimicMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "test")
        elif self.dset_name == "iuxray":
            # 加载IU X-ray数据集
            self.train_set = IUXrayMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "train")
            self.val_set = IUXrayMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "val")
            self.test_set = IUXrayMultiModal(self.config.dataset.img_path, self.config.dataset.ann_path, self.config.dataset.view, "test")

        # 创建数据加载器
        self.train_loader = DataLoader(self.train_set, batch_size=self.config.dataloader.batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.val_loader = DataLoader(self.val_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.config.dataloader.eval_batch_size, shuffle=True, num_workers= self.config.dataloader.num_workers, pin_memory=True, drop_last=False)
        print("------------------------------Data Loaded Successfully-------------------------")

    def load_model(self):
        """加载模型和相关组件"""
        self.model = get_mmclf(config=self.config.model)
        self.tokenizer = get_tokenizer(config=self.config.model)
        self.criterion = get_criterion(self.config.criterion.name, self.config.criterion)
        self.optimizer = get_optimizer(self.config.optimizer.name, self.model.parameters(), self.config.optimizer)
        self.grad_scaler =  torch.cuda.amp.GradScaler()
        print("------------------------------Model Loaded Successfully-------------------------")

    def save_best(self, comms):
        """保存最佳模型"""
        ckpt_path = os.path.join(self.save_dir, f"model_best.pth")
        torch.save({"net":self.model.state_dict(), "comms":comms}, ckpt_path)

    def load_best(self):
        """加载最佳模型"""
        ckpt_path = os.path.join(self.save_dir, f"model_best.pth")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["net"])
        print(f"Best Model is at comms : {checkpoint['comms']}")

    def save_log(self):
        """保存验证集AUC记录"""
        log_path = os.path.join(self.save_dir, "val_aucs.pkl")
        with open(log_path, "wb") as f:
            pickle.dump(self.val_track, f)

    def run_standalone(self):
        """独立训练模式"""
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("----------------Standalone Training-------------------------")
        self.val_auc = 0
        self.model.cuda()
        for i in range(self.config.train.total_epoch):
            print(f"Server: Epoch {i}")
            self.train_epoch()
            cur_auc = self.val()
            self.val_track.append(cur_auc)
            if cur_auc > self.val_auc:
                self.val_auc = cur_auc
                self.save_best(i)
            self.cur_epoch+=1
        print("------------------------------------------------------------")
        self.save_log()
        self.load_best()
        self.test()

    def test(self):
        """测试模型性能"""
        self.model.cuda()
        self.model.eval()
        test_evaluator = MetricCollection({
            "AUC":  MultilabelAUROC(num_labels=14, average="macro", thresholds=None),
            "AUCperLabel":  MultilabelAUROC(num_labels=14, average="none", thresholds=None)
        })
        with tqdm(self.test_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                test_evaluator.update(output["logits"], label.long())
        metrics = test_evaluator.compute()
        print(f"AUC : {metrics['AUC']}")
        print(f"AUCperLabel : {metrics['AUCperLabel']}")
        self.wandb.log({"Test AUC(Aggregrated)":metrics['AUC'].item()})
        self.evaluator.reset()

    def run(self, comms):
        """联邦学习训练模式"""
        self.model.cuda()
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        print("------------------------------------------------------------")
        for i in range(self.config.train.local_epoch):
            print(f"Server:-  Comm round{comms} local_epoch:{self.cur_epoch}  round_epoch: {i}")
            self.train_epoch()
            self.cur_epoch +=1
            print("------------------------------------------------------------")
        self.model.cpu()
        import gc
        gc.collect()

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        print("Training Model:")
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                self.optimizer.zero_grad()
                images = frames.cuda()
                label = label.cuda()
                # 使用混合精度训练
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model(self.tokenizer, images, text)
                    loss = self.criterion(output["logits"], label)

                self.grad_scaler.scale(loss).backward()

                # 梯度裁剪
                if self.config.train.grad_clip > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(),
                                                   self.config.train.grad_clip)

                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tepoch.set_postfix(Loss=loss.item())

    def val(self):
        """验证模型性能"""
        self.model.eval()
        print('Validating Model:')
        with tqdm(self.val_loader, unit="batch") as tepoch:
            for frames, label, text, _ in tepoch:
                images = frames.cuda()
                label = label.cuda()
                with torch.no_grad():
                    output = self.model(self.tokenizer, images, text)
                self.evaluator.update(output["logits"], label.long())
        metrics = self.evaluator.compute()
        print(f"Val AUC : {metrics['AUC']}")
        if self.wandb:
            self.wandb.log({"Val AUC(Server)":metrics['AUC'].item()}, step=self.cur_epoch)
        self.evaluator.reset()
        return metrics['AUC'].item()
