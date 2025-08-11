import argparse
import os
import torch
from exp.exp_AQP_train import Exp_AQP_Forecast
import random
import numpy as np
import warnings
import sys
from utils.utils import Logger
from datetime import datetime

warnings.filterwarnings('ignore')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    fix_seed = 1234
    set_seed(fix_seed)

    parser = argparse.ArgumentParser(description='AirPCM')
    # basic config
    parser.add_argument('--model', type=str, default='AirPCM')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/Mydata/', help='location of model checkpoints')
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
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')

    # patching
    parser.add_argument('--patch_size', type=int, default=6)  # 最好是(6,4)
    parser.add_argument('--stride', type=int, default=4)

    args = parser.parse_args()

    setting = '{}_{}_sl{}_pl{}_bs{}_pt_{}_sr_{}_{}'.format(
        args.model,
        args.data,
        args.seq_len,
        args.pred_len,
        args.batch_size,
        args.patch_size,
        args.stride,
        # datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        datetime.now().strftime("%Y-%m-%d")
    )

    sys.stdout = Logger("log/train/" + setting + ".txt", sys.stdout)

    if torch.cuda.is_available():
        args.device = torch.device('cuda:%d' % args.device)
    else:
        args.device = torch.device('cpu')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('Args in experiment:')
    print(args)

    Exp = Exp_AQP_Forecast

    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)
    exp = Exp(args)
    exp.train(setting)
    sys.stdout.flush()
