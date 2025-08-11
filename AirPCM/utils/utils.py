import os
import torch
import numpy as np
import sys


def load_graph_data(graph_path):
    data = np.load(graph_path)

    adj_mx = data['adj_mx']
    edge_index = data['edge_index']  # 边的索引
    edge_attr = data['edge_attr']  # 边的属性
    node_attr = data['node_attr']  # 节点的属性

    return adj_mx, edge_index.T, edge_attr, node_attr


def exchange_df_column(df, col1, col2):
    assert (col1 in df.columns) and (col2 in df.columns)
    df[col1], df[col2] = df[col2].copy(), df[col1].copy()
    df = df.rename(columns={col1: 'temp', col2: col1})
    df = df.rename(columns={'temp': col2})
    return df


def MAE(pred, true):
    return torch.abs(pred - true)


def MSE(pred, true):
    return (pred - true) ** 2


def MAPE(pred, true):
    return torch.abs((pred - true) / true)


def SMAPE(pred, true):
    # Avoid division by zero by adding a small constant
    denominator = (torch.abs(true) + torch.abs(pred)) / 2 + 1e-8
    # Calculate the SMAPE
    smape_value = torch.mean(torch.abs(pred - true) / denominator)
    return smape_value


def masked_loss(y_pred, y_true, loss_func):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()  # assign the sample weights of zeros to nonzero-values
    loss = loss_func(y_pred, y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_rmse_loss(y_pred, y_true):
    y_true[y_true < 1e-4] = 0
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.pow(y_pred - y_true, 2)
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())


def compute_all_metrics(y_pred, y_true):
    mae = masked_loss(y_pred, y_true, MAE).item()
    rmse = masked_rmse_loss(y_pred, y_true).item()
    smape = masked_loss(y_pred, y_true, SMAPE).item()
    return mae, smape, rmse


def sudden_changes_mask(labels, datapath, null_val=np.nan, threshold_start=75, threshold_change=20):

    b, t, n = labels.shape
    mask = torch.zeros(size=(b, t, n))
    mask_ones = torch.ones(size=(b, n))
    mask_zeros = torch.zeros(size=(b, n))
    for t in range(1, 24):
        prev = labels[:, t - 1]
        curr = labels[:, t]
        mask[:, t] = torch.where((torch.BoolTensor(curr > threshold_start)), mask_ones, mask[:, t])
        mask[:, t] = torch.where(torch.abs(torch.Tensor(curr - prev)) > threshold_change, mask_ones, mask[:, t])
        if not np.isnan(null_val):
            mask[:, t] = torch.where(torch.BoolTensor(prev < null_val + 0.1), mask_zeros, mask[:, t])
        else:
            mask[:, t] = torch.where(torch.isnan(curr), mask_zeros, mask[:, t])
    return mask


def masked_mae(preds, labels, null_val=np.nan, mask=None):
    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds, labels, null_val=np.nan, mask=None):
    '''
    Calculate MSE.
    The missing values in labels will be masked.
    '''
    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan, mask=None):
    if mask == None:
        return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
    else:
        return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, mask=mask))

def masked_smape(preds, labels, null_val=np.nan, mask=None):
    '''
    Calculate SMAPE (Symmetric Mean Absolute Percentage Error).
    The missing values in labels will be masked.
    '''
    if mask is None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)

    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    absolute_error = torch.abs(preds - labels)
    denominator = (torch.abs(preds) + torch.abs(labels)) / 2
    denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    smape_loss = absolute_error / denominator
    smape_loss = smape_loss * mask
    smape_loss = torch.where(torch.isnan(smape_loss), torch.zeros_like(smape_loss), smape_loss)
    return torch.mean(smape_loss)

def compute_sudden_change(mask, pred, real, null_value):
    mae = masked_mae(pred, real, null_value, mask).item()
    rmse = masked_rmse(pred, real, null_value, mask).item()
    smape = masked_smape(pred, real, null_value, mask).item()
    return mae, rmse, smape


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass
