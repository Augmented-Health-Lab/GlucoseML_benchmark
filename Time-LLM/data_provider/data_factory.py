from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_M4
)
from torch.utils.data import DataLoader

# >>> NEW imports <<<
from data_provider.glucose_dataset import MultiSubjectGlucoseDataset
from data_provider.samplers import SubjectBalancedSampler
from data_provider.samplers import SubsampleWindowSampler



data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,

}


def data_provider(args, flag):
    """
    flag: 'train' | 'val' | 'test'
    """
    sampler = None
    shuffle_flag = True

    # ===============================
    # ✅ Glucose dataset (NEW branch)
    # ===============================
    if args.data == 'Glucose':
        # =========================
        # TRAINING MODE
        # =========================
        if args.is_training == 1:
            assert flag in ['train', 'val'], \
                "Training mode only supports train / val splits for Glucose"

            data_dir = args.root_path

        # =========================
        # TEST-ONLY MODE
        # =========================
        else:
            assert flag == 'test', \
                "Test-only mode should only request test split"

            assert args.test_root_path is not None, \
                "For Glucose test split, please provide --test_root_path"

            data_dir = args.test_root_path

        data_set = MultiSubjectGlucoseDataset(
            data_dir=data_dir,
            split=flag,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            stride=args.stride,
            scale_value=args.scale_value
        )

        # if flag == 'train':
        #     sampler = SubjectBalancedSampler(data_set)
        #     data_loader = DataLoader(
        #         data_set,
        #         batch_size=args.batch_size,
        #         sampler=sampler,              # <<< key point
        #         num_workers=args.num_workers,
        #         drop_last=True
        #     )
        # else:
        #     data_loader = DataLoader(
        #         data_set,
        #         batch_size=args.batch_size,
        #         shuffle=False,                 # <<< no shuffle
        #         num_workers=args.num_workers,
        #         drop_last=False
        #     )
        if flag == 'train' and args.max_windows_per_epoch > 0:
            sampler = SubsampleWindowSampler(
                data_set,
                max_windows_per_epoch=args.max_windows_per_epoch,
                seed=args.seed
            )
            shuffle_flag = False   # sampler & shuffle 不能同时用

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle_flag if sampler is None else False,
            num_workers=args.num_workers,
            drop_last=(flag == 'train')
        )

        return data_set, data_loader

    # ======================================================
    # 🔁 Original Time-LLM datasets (unchanged)
    # ======================================================
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
