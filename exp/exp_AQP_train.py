import sys
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from models import AirPMC
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils.utils import load_graph_data

warnings.filterwarnings('ignore')


class Exp_AQP_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.model = AirPMC.Model(self.args).to(self.device)

        # 下载图数据
        graph_path = os.path.join(args.root_path, "graph_data_" + self.args.data + ".npz")
        adj_mx, edge_index, edge_attr, node_attr = load_graph_data(graph_path)
        self.adj_mx = adj_mx  # N x N
        self.edge_index = edge_index  # adjacent list: 2 x M
        self.edge_attr = edge_attr  # M x D
        self.node_attr = node_attr  # N x D
        # 转换为 PyTorch 张量
        self.adj_mx = torch.from_numpy(self.adj_mx).float().to(self.device)
        self.edge_attr = torch.from_numpy(self.edge_attr).float().to(self.device)
        self.node_attr = torch.from_numpy(self.node_attr).float().to(self.device)
        self.edge_index = torch.from_numpy(self.edge_index).to(self.device)

        self.train_data, self.train_loader = self._get_data(flag='train')  # 训练或者验证数据集的dataloader
        self.vali_data, self.vali_loader = self._get_data(flag='val')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, eps=1e-8)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                                       patience=3,
                                                                       verbose=True)
        self.criterion = self._select_criterion()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self):
        if self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'MAE':
            criterion = nn.L1Loss()
        elif self.args.loss == 'Huber':
            criterion = nn.HuberLoss(reduction='mean', delta=1.0)
        return criterion

    @torch.no_grad()
    def vali(self, vali_loader):
        total_loss = []
        self.model.eval()
        for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader), total=len(vali_loader),
                                          desc="Validation", unit="batch"):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            outputs = self.model(batch_x, self.edge_index, self.edge_attr)
            # y = batch_y[:, :, :, 0:self.args.pred_variable].float().to(self.device)

            y_hat = outputs[:, :, :, 0].float().to(self.device)
            y = batch_y[:, :, :, 0].float().to(self.device)

            pred = y_hat.detach().cpu()
            true = y.detach().cpu()

            loss = self.criterion(pred, true)  # 不再计算相似度损失
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in tqdm(range(self.args.train_epochs),
                          total=self.args.train_epochs,
                          desc="Training Progress",
                          unit="epoch",
                          leave=True,
                          ncols=100,
                          dynamic_ncols=True,
                          ):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y) in tqdm(enumerate(self.train_loader),
                                              total=len(self.train_loader), desc="Training",
                                              unit="batch", leave=True):
                iter_count += 1
                self.optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x, self.edge_index, self.edge_attr)
                y = batch_y[:, :, :, 0:self.args.pred_variable].float().to(self.device)

                pred_loss = self.criterion(outputs, y)
                loss = pred_loss

                train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(self.vali_loader)  # 验证
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".
                  format(epoch + 1, train_steps, train_loss, vali_loss))

            self.lr_scheduler.step(vali_loss)
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
