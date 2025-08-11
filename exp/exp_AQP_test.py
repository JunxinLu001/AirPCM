from data_provider.data_factory import data_provider
import torch
from models import AirPCM
import warnings
import os
from tqdm import tqdm
import numpy as np
from utils.utils import load_graph_data, compute_all_metrics, sudden_changes_mask, compute_sudden_change

warnings.filterwarnings('ignore')


class Exp_AQP_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.path = args.root_path

        self.device = self.args.device
        self.model = AirPCM.Model(self.args).to(self.device)

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

        self.test_data, self.test_loader = self._get_data(flag='test')
        self.inverse_transform = self.test_data.inverse_transform

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def test(self, setting, checkpoint_path):

        print('loading model')
        self.model.load_state_dict(torch.load(checkpoint_path))

        with torch.no_grad():
            self.model.eval()

            truths = []
            preds = []

            for i, (batch_x, batch_y) in tqdm(enumerate(self.test_loader),
                                              total=len(self.test_loader),
                                              desc="Testing",
                                              unit="batch"):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                pred, attn = self.model(batch_x, self.edge_index, self.edge_attr)
                y_hat = pred[:, :, :, 0].float().to(self.device)
                y = batch_y[:, :, :, 0].float().to(self.device)
                preds.append(y_hat.cpu())
                truths.append(y.cpu())

            truths = torch.cat(truths, dim=0)
            preds = torch.cat(preds, dim=0)

            all_mae = []
            all_smape = []
            all_rmse = []

            for i in range(0, self.args.pred_len, 8):
                pred = preds[:, i: i + 8]
                truth = truths[:, i: i + 8]
                mae, smape, rmse = self._compute_loss_eval(truth, pred)

                all_mae.append(mae)
                all_smape.append(smape)
                all_rmse.append(rmse)
                print('Evaluation {}h-{}h: - mae - {:.4f} - rmse - {:.4f} - mape - {:.4f}'.format(
                    i * 3, (i + 8) * 3, mae, rmse, smape))
            # three days
            mae, smape, rmse = self._compute_loss_eval(truths, preds)
            all_mae.append(mae)
            all_smape.append(smape)
            all_rmse.append(rmse)

            truths = self.inverse_transform(truths)
            preds = self.inverse_transform(preds)

            mask_sudden_change = sudden_changes_mask(truths, datapath=self.path, null_val=0.0,
                                                     threshold_start=75, threshold_change=20)
            sc_mae, sc_rmse, sc_smape = compute_sudden_change(mask_sudden_change.squeeze(-1), preds, truths,
                                                              null_value=0.0)
            all_mae.append(sc_mae)
            all_smape.append(sc_smape)
            all_rmse.append(sc_rmse)

            print('Evaluation all: - mae - {:.4f} - rmse - {:.4f} - mape - {:.4f} '
                  '- sc_mae - {:.4f} - sc_rmse - {:.4f} - sc-mape - {:.4f}'
                  .format(mae, rmse, smape, sc_mae, sc_rmse, sc_smape))
            return all_mae, all_smape, all_rmse

    def _compute_loss_eval(self, y_true, y_predicted):
        y_true = self.inverse_transform(y_true)
        y_predicted = self.inverse_transform(y_predicted)
        return compute_all_metrics(y_predicted, y_true)
