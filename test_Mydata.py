import argparse
import os
import torch
from exp.exp_AQP_test import Exp_AQP_Forecast
import random
import numpy as np
import warnings
from datetime import datetime
import sys
from utils.utils import Logger
warnings.filterwarnings('ignore')


def get_mean_std(data_list):
    return data_list.mean(), data_list.std()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fix_seed = 1234
set_seed(fix_seed)

parser = argparse.ArgumentParser(description='AirPCM')
# basic config
parser.add_argument('--model', type=str, default='AirPCM')
parser.add_argument('--device', type=int, default=3, help='gpu number')

# data loader
parser.add_argument('--root_path', type=str, default='./dataset/Mydata/', help='root path of the data file')
parser.add_argument('--data', type=str, default='Mydata', help='dataset type')
parser.add_argument('--number_variable', type=int, default=11, help='number of variable')
parser.add_argument('--pred_variable', type=int, default=6, help='number of variable')
parser.add_argument('--station_num', type=int, default=453, help='number of station')
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

# model define
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--hidden_channels', type=int, default=128, help='hidden_channels of model')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--embed', type=bool, default=False)
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--interval', type=str, default='3h', help='interval')
parser.add_argument('--head', type=int, default=8, help='attention head')
parser.add_argument('--lag_window', type=int, default=8, help='attention head')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')  # 实验的运行次数
parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')

# patching
parser.add_argument('--patch_size', type=int, default=6)
parser.add_argument('--stride', type=int, default=4)

args = parser.parse_args()
if torch.cuda.is_available():
    args.device = torch.device('cuda:%d' % args.device)
else:
    args.device = torch.device('cpu')

setting = '{}_{}_sl{}_pl{}_bs{}_pt_{}_sr_{}_{}'.format(
    args.model,
    args.data,
    args.seq_len,
    args.pred_len,
    args.batch_size,
    args.patch_size,
    args.stride,
    datetime.now().strftime("%Y-%m-%d")
)
sys.stdout = Logger("log/test/" + setting + ".txt", sys.stdout)
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Args in experiment:')
print(args)
checkpoints_path = ''  # 放置保存的模型路径

rmse_list, mae_list, mape_list = [], [], []

for exp_idx in range(args.itr):
    args.exp_idx = exp_idx
    print('\nNo%d experiment ~~~' % exp_idx)

    exp = Exp_AQP_Forecast(args)
    mae, mape, rmse = exp.test(setting, checkpoints_path)
    mae_list.append(mae)
    mape_list.append(mape)
    rmse_list.append(rmse)

mae_list = np.array(mae_list)
mape_list = np.array(mape_list)
rmse_list = np.array(rmse_list)

seq_len = [(0, 8), (8, 16), (16, 24)]
output_text = ''
output_text += '--------- CauAir Final Results ------------\n'
for i, (start, end) in enumerate(seq_len):
    output_text += 'Evaluation seq {}h-{}h:\n'.format(start, end)
    output_text += 'MAE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mae_list[:, i])[0],
                                                             get_mean_std(mae_list[:, i])[1])
    output_text += 'SMAPE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mape_list[:, i])[0],
                                                               get_mean_std(mape_list[:, i])[1])
    output_text += 'RMSE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(rmse_list[:, i])[0],
                                                              get_mean_std(rmse_list[:, i])[1])

output_text += 'ALL: MAE: {:.4f}, RMSE: {:.4f}, SMAPE: : {:.4f}\n'.format(get_mean_std(mae_list[:, 3])[0],
                                                                          get_mean_std(rmse_list[:, 3])[0],
                                                                          get_mean_std(mape_list[:, 3])[0])

output_text += 'ALL: SC_MAE: {:.4f}, SC_RMSE: {:.4f}, SC_SMAPE: : {:.4f}\n\n'.format(get_mean_std(mae_list[:, 4])[0],
                                                                                     get_mean_std(rmse_list[:, 4])[0],
                                                                                     get_mean_std(mape_list[:, 4])[0])

folder_path = './test_results/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
with open(folder_path + args.data + '_results.txt', 'a') as file:
    file.write(output_text)
print("\n\n\n\n")