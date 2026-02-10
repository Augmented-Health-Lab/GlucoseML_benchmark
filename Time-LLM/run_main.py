import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from collections import defaultdict
import csv

from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import sys
import json
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument(
    '--scale_value',
    type=float,
    default=100.0,
    help='scaling factor for glucose values'
)
parser.add_argument(
    "--max_windows_per_epoch",
    type=int,
    default=-1,
    help="Subsample training windows per epoch (-1 = use all)"
)
parser.add_argument(
    '--test_root_path',
    type=str,
    default=None,
    help='root path for glucose test dataset'
)
parser.add_argument(
    "--patch_stride",
    type=int,
    default=3,
    help="Sliding window stride for glucose dataset"
)

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision='no')

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    if args.is_training == 1:
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = None, None
    else:
        train_data, train_loader = None, None
        vali_data, vali_loader = None, None
        test_data, test_loader = data_provider(args, 'test')


    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    accelerator.print(
        f"[LLM] model={args.llm_model}, layers={args.llm_layers}, frozen=True"
    )


    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    # train_steps = len(train_loader)
    # early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    # trained_parameters = []
    # for p in model.parameters():
    #     if p.requires_grad is True:
    #         trained_parameters.append(p)

    # model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    # if args.lradj == 'COS':
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    # else:
    #     scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
    #                                         steps_per_epoch=train_steps,
    #                                         pct_start=args.pct_start,
    #                                         epochs=args.train_epochs,
    #                                         max_lr=args.learning_rate)

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    # train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        # train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    
    # =====================================================
    # TEST-ONLY MODE (skip training loop entirely)
    # =====================================================
    if args.is_training == 0:
        accelerator.print(">>> Running TEST-ONLY evaluation")
        test_data, test_loader = data_provider(args, 'test')

        # -------- load best checkpoint --------
        best_ckpt = os.path.join(path, "checkpoint")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")

        accelerator.print(f">>> Loading checkpoint from {best_ckpt}")
        model.load_state_dict(
            torch.load(best_ckpt, map_location=accelerator.device)
        )
        model = accelerator.prepare(model)
        model.eval()

        # -------- run test evaluation --------
        test_loss, test_mae, preds, trues, subject_ids = vali(
            args,
            accelerator,
            model,
            test_data,
            test_loader,
            criterion,
            mae_metric,
            return_preds=True,
            return_subject_ids=True,
        )
        # =====================================================
        # Horizon-wise MAE / RMSE (Glucose-specific evaluation)
        # =====================================================
        scale = args.scale_value       # e.g. 100.0
        step_minutes = 5               # 5-min sampling

        horizons = [15, 30, 60, 90]

        # -----------------------------
        # Aggregate per-subject errors
        # -----------------------------
        per_subj_errs = defaultdict(list)  # sid -> list of (pred_len,) arrays
        for i, sid in enumerate(subject_ids):
            err = preds[i, :, 0] - trues[i, :, 0]   # scaled space
            per_subj_errs[sid].append(err)

        # -----------------------------
        # Prepare output folder
        # -----------------------------
        save_dir = os.path.join(path, "test_metrics")
        os.makedirs(save_dir, exist_ok=True)

        # -----------------------------
        # Save per-subject overall MAE/RMSE
        # -----------------------------
        csv_path = os.path.join(save_dir, "per_subject_overall.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_id", "MAE_mgdl", "RMSE_mgdl", "num_windows"])
            for sid, errs_list in per_subj_errs.items():
                errs = np.concatenate(errs_list, axis=0)  # flatten all windows & all horizons
                mae = np.mean(np.abs(errs)) * scale
                rmse = np.sqrt(np.mean(errs ** 2)) * scale
                writer.writerow([sid, float(mae), float(rmse), len(errs_list)])

        accelerator.print(f"[TEST] Saved per-subject overall metrics -> {csv_path}")

        # -----------------------------
        # Save per-subject horizon-wise MAE/RMSE
        # -----------------------------
        csv_path2 = os.path.join(save_dir, "per_subject_horizons.csv")
        with open(csv_path2, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["subject_id", "num_windows"]
            for h in horizons:
                header += [f"MAE_{h}min_mgdl", f"RMSE_{h}min_mgdl"]
            writer.writerow(header)

            for sid, errs_list in per_subj_errs.items():
                errs_stack = np.stack(errs_list, axis=0)  # (num_windows, pred_len)

                row = [sid, len(errs_list)]
                for h in horizons:
                    idx = h // step_minutes - 1
                    if idx < 0 or idx >= errs_stack.shape[1]:
                        row += ["", ""]
                        continue
                    e = errs_stack[:, idx]  # (num_windows,)
                    mae_h = np.mean(np.abs(e)) * scale
                    rmse_h = np.sqrt(np.mean(e ** 2)) * scale
                    row += [float(mae_h), float(rmse_h)]
                writer.writerow(row)

        accelerator.print(f"[TEST] Saved per-subject horizon metrics -> {csv_path2}")


        accelerator.print(
        f"[TEST] MSE: {test_loss:.6f}, MAE: {test_mae:.6f}"
        )

        sys.exit(0)  # 🔴 VERY IMPORTANT: stop here, do NOT enter training loop
    train_steps = len(train_loader)

    early_stopping = EarlyStopping(
        accelerator=accelerator,
        patience=args.patience
    )

    trained_parameters = [
        p for p in model.parameters() if p.requires_grad
    ]

    model_optim = optim.Adam(
        trained_parameters,
        lr=args.learning_rate
    )

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim, T_max=20, eta_min=1e-8
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate
        )

    # prepare for accelerate (TRAIN ONLY)
    train_loader, vali_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, model, model_optim, scheduler
    )
    if args.is_training == 1:
        assert args.test_root_path is None, \
            "Training mode should NOT use test_root_path"

    


    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, batch in tqdm(enumerate(train_loader)):
            if len(batch) == 4:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                subject_ids = None
            elif len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, subject_ids = batch
            else:
                raise ValueError(f"Unexpected batch size from dataloader: {len(batch)}")

            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
        #     iter_count += 1
        #     model_optim.zero_grad()

        #     batch_x = batch_x.float().to(accelerator.device)
        #     batch_y = batch_y.float().to(accelerator.device)
        #     batch_x_mark = batch_x_mark.float().to(accelerator.device)
        #     batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        # vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
        # test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
        # accelerator.print(
        #     "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
        #         epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))
        vali_loss, vali_mae_loss = vali(
            args, accelerator, model, vali_data, vali_loader, criterion, mae_metric
        )

        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} | Val MAE: {2:.7f}".format(
                epoch + 1, train_loss, vali_mae_loss
            )
        )

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

# accelerator.wait_for_everyone()
# if accelerator.is_local_main_process:
#     path = './checkpoints'  # unique checkpoint saving path
#     del_files(path)  # delete checkpoint files
#     accelerator.print('success delete checkpoints')