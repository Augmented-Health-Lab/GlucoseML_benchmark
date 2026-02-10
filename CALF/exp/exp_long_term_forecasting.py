from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.cmLoss import cmLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import csv


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, vali_test=False):
        data_set, data_loader = data_provider(self.args, flag, vali_test)
        return data_set, data_loader

    def _select_optimizer(self):
        param_dict = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": self.args.learning_rate}
        ]
        model_optim = optim.Adam([param_dict[1]], lr=self.args.learning_rate)
        loss_optim = optim.Adam([param_dict[0]], lr=self.args.learning_rate)

        return model_optim, loss_optim

    def _select_criterion(self):
        criterion = cmLoss(self.args.feature_loss, 
                           self.args.output_loss, 
                           self.args.task_loss, 
                           self.args.task_name, 
                           self.args.feature_w, 
                           self.args.output_w, 
                           self.args.task_w)
        return criterion

    # def train(self, setting):
    #     train_data, train_loader = self._get_data(flag='train')
    #     vali_data, vali_loader = self._get_data(flag='val')
    #     test_data, test_loader = self._get_data(flag='test', vali_test=True)

    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     time_now = time.time()

    #     train_steps = len(train_loader)
    #     early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    #     model_optim, loss_optim = self._select_optimizer()
    #     criterion = self._select_criterion()
        
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

    #     for epoch in range(self.args.train_epochs):
    #         iter_count = 0
    #         train_loss = []

    #         self.model.train()
    #         epoch_time = time.time()
    #         for i, batch in enumerate(train_loader):
    #             if len(batch) == 4:
    #                 batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    #                 subject_ids = None
    #             elif len(batch) == 5:
    #                 batch_x, batch_y, batch_x_mark, batch_y_mark, subject_ids = batch
    #             else:
    #                 raise ValueError(f"Unexpected batch size: {len(batch)}")

    #             iter_count += 1
    #             model_optim.zero_grad()
    #             loss_optim.zero_grad()

    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
                
    #             outputs_dict = self.model(batch_x)
                
    #             loss = criterion(outputs_dict, batch_y)

    #             train_loss.append(loss.item())

    #             if (i + 1) % 100 == 0:
    #                 print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                 speed = (time.time() - time_now) / iter_count
    #                 left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #                 print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                 iter_count = 0
    #                 time_now = time.time()

    #             loss.backward()
    #             model_optim.step()
    #             loss_optim.step()

    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         train_loss = np.average(train_loss)

    #         vali_loss = self.vali(vali_data, vali_loader, criterion)
    #         test_loss = self.vali(test_data, test_loader, criterion)

    #         print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
    #             epoch + 1, train_steps, train_loss, vali_loss, test_loss))

    #         if self.args.cos:
    #             scheduler.step()
    #             print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
    #         else:
    #             adjust_learning_rate(model_optim, epoch + 1, self.args)

    #         early_stopping(vali_loss, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path))

    #     return self.model
    def train(self, setting):
        # ===============================
        # Load TRAIN + VAL only
        # ===============================
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True
        )

        model_optim, loss_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim,
            T_max=self.args.tmax,
            eta_min=1e-8
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                # -------- batch unpack --------
                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                elif len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                iter_count += 1
                model_optim.zero_grad()
                loss_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # -------- forward --------
                outputs_dict = self.model(batch_x)
                loss = criterion(outputs_dict, batch_y)

                train_loss.append(loss.item())

                # -------- backward --------
                loss.backward()
                model_optim.step()
                loss_optim.step()

                # -------- logging --------
                if (i + 1) % 100 == 0:
                    print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}"
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            print(
                f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s"
            )
            train_loss = np.average(train_loss)

            # ===============================
            # VALIDATION ONLY
            # ===============================
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print(
                f"Epoch: {epoch + 1} | "
                f"Train Loss: {train_loss:.7f} | "
                f"Val Loss: {vali_loss:.7f}"
            )

            # ===============================
            # LR scheduling
            # ===============================
            if self.args.cos:
                scheduler.step()
                print(f"lr = {model_optim.param_groups[0]['lr']:.10f}")
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            # ===============================
            # Early stopping
            # ===============================
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # ===============================
        # Load best checkpoint
        # ===============================
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    

    # def vali(self, vali_data, vali_loader, criterion):
    #     total_loss = []

    #     self.model.in_layer.eval()
    #     self.model.out_layer.eval()
    #     self.model.time_proj.eval()
    #     self.model.text_proj.eval()

    #     with torch.no_grad():
    #         # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
    #         for i, batch in enumerate(vali_loader):
    #             if len(batch) == 4:
    #                 batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    #                 subject_ids = None
    #             elif len(batch) == 5:
    #                 batch_x, batch_y, batch_x_mark, batch_y_mark, subject_ids = batch
    #             else:
    #                 raise ValueError(f"Unexpected batch size: {len(batch)}")

    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)

    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             outputs = self.model(batch_x)

    #             outputs_ensemble = outputs['outputs_time']
    #             # encoder - decoder
    #             outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
    #             batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

    #             pred = outputs_ensemble.detach().cpu()
    #             true = batch_y.detach().cpu()

    #             loss = F.mse_loss(pred, true)

    #             total_loss.append(loss)

    #     total_loss = np.average(total_loss)

    #     self.model.in_layer.train()
    #     self.model.out_layer.train()
    #     self.model.time_proj.train()
    #     self.model.text_proj.train()

    #     return total_loss
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []

        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                # -------- batch unpack --------
                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                elif len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                outputs_ensemble = outputs['outputs_time']

                outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                loss = F.mse_loss(outputs_ensemble, batch_y)
                total_loss.append(loss.item())

        self.model.train()
        return float(np.mean(total_loss))

    def test(self, setting, test=0):
        # -----------------------------
        # Zero-shot handling (unchanged)
        # -----------------------------
        if self.args.zero_shot:
            self.args.data = self.args.target_data
            self.args.data_path = f"{self.args.data}.csv"

        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'),
                          map_location=self.device)
            )

        # -----------------------------
        # Containers
        # -----------------------------
        preds = []
        trues = []

        per_subject_preds = defaultdict(list)
        per_subject_trues = defaultdict(list)

        folder_path = os.path.join('./test_results/', setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # -------- unpack batch (TimeLLM-compatible) --------
                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    subject_ids = None
                elif len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, subject_ids = batch
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # -------- forward --------
                outputs = self.model(batch_x[:, -self.args.seq_len:, :])
                outputs_ensemble = outputs['outputs_time']
                outputs_ensemble = outputs_ensemble[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                pred = outputs_ensemble.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                # -------- subject-aware aggregation --------
                if subject_ids is not None:
                    for j, sid in enumerate(subject_ids):
                        per_subject_preds[sid].append(pred[j])
                        per_subject_trues[sid].append(true[j])

                # -------- optional visualization (unchanged) --------
                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_np[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f"{i}.pdf"))

        # -----------------------------
        # Stack global predictions
        # -----------------------------
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # -----------------------------
        # Global metrics (CALF default)
        # -----------------------------
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"[TEST] mse:{mse}, mae:{mae}")

        result_path = os.path.join('./results/', setting)
        os.makedirs(result_path, exist_ok=True)

        np.save(os.path.join(result_path, 'metrics.npy'),
                np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(result_path, 'pred.npy'), preds)
        np.save(os.path.join(result_path, 'true.npy'), trues)

        # -----------------------------
        # Per-subject evaluation (Glucose)
        # -----------------------------
        if len(per_subject_preds) > 0:
            save_dir = os.path.join(result_path, "per_subject_metrics")
            os.makedirs(save_dir, exist_ok=True)

            scale = getattr(self.args, "scale_value", 1.0)
            step_minutes = 5
            horizons = [15, 30, 60, 90]

            # ---- overall per-subject ----
            csv_path = os.path.join(save_dir, "per_subject_overall.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["subject_id", "MAE_mgdl", "RMSE_mgdl", "num_windows"])
                for sid in per_subject_preds:
                    p = np.concatenate(per_subject_preds[sid], axis=0)
                    t = np.concatenate(per_subject_trues[sid], axis=0)
                    err = p - t
                    mae_s = np.mean(np.abs(err)) * scale
                    rmse_s = np.sqrt(np.mean(err ** 2)) * scale
                    writer.writerow([sid, float(mae_s), float(rmse_s), len(per_subject_preds[sid])])

            # ---- horizon-wise per-subject ----
            csv_path2 = os.path.join(save_dir, "per_subject_horizons.csv")
            with open(csv_path2, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["subject_id", "num_windows"]
                for h in horizons:
                    header += [f"MAE_{h}min_mgdl", f"RMSE_{h}min_mgdl"]
                writer.writerow(header)

                for sid in per_subject_preds:
                    p = np.stack(per_subject_preds[sid], axis=0)
                    t = np.stack(per_subject_trues[sid], axis=0)
                    err = p - t

                    row = [sid, err.shape[0]]
                    for h in horizons:
                        idx = h // step_minutes - 1
                        if idx < 0 or idx >= err.shape[1]:
                            row += ["", ""]
                        else:
                            e = err[:, idx]
                            row += [
                                float(np.mean(np.abs(e)) * scale),
                                float(np.sqrt(np.mean(e ** 2)) * scale),
                            ]
                    writer.writerow(row)

            print(f"[TEST] Per-subject metrics saved to {save_dir}")

        return
