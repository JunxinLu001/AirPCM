from data_provider.data_loader import Dataset_Mydata
from torch.utils.data import DataLoader

data_dict = {
    'Mydata': Dataset_Mydata,
}


def data_provider(args, flag):

    Data = data_dict[args.data]

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
    else:
        shuffle_flag = False
        drop_last = False
    batch_size = args.batch_size

    if args.data == "Beijing":
        args.normalized_columns = [0, 1, 2, 3, 4, 5]
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            scale=True,
            normalized_col=args.normalized_columns,
        )
    elif args.data == "Beijing_ALL":
        args.normalized_columns = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2",
                                   "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
        # args.normalized_columns = ["PM2.5" ,
        #                            "temperature", "pressure", "humidity", "wind_speed", "wind_direction"]
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            patch_size=args.patch_size,
            freq=args.interval,
            embed=args.embed,
            scale=True,
            normalized_col=args.normalized_columns
        )
    elif args.data == "America":
        args.normalized_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            freq=args.interval,
            embed=args.embed,
            scale=True,
            normalized_col=args.normalized_columns
        )
    elif args.data == "Mydata":
        # args.normalized_columns = [0, 1, 2, 3, 4, 5]
        args.normalized_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            freq=args.interval,
            embed=args.embed,
            scale=True,
            normalized_col=args.normalized_columns
        )

    elif args.data == "Mydata_D":
        args.normalized_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            freq=args.interval,
            embed=args.embed,
            scale=True,
            normalized_col=args.normalized_columns
        )
    else:
        args.normalized_columns = [0, 1, 2, 3, 4, 5]
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            freq=args.interval,
            embed=args.embed,
            scale=True,
            normalized_col=args.normalized_columns
        )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader