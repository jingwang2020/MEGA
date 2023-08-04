# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 20:28
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : save_plot_data.py
# @Software: PyCharm
from DataProcess.dataset import MuliTsBatchedWindowDataset
from torch.utils.data import DataLoader
import torch
from tqdm.std import tqdm
import numpy as np
import torch.nn as nn
import pandas as pd
from .evatuate import adjust_predicts
from sklearn.metrics._ranking import _binary_clf_curve


def save_train_test(model, train_data, test_data, test_label, device=None, args=None):
    """
    :param model:
    :param train_data:
    :param test_data:
    :param test_label:
    :param device:
    :param args:
    :return:
    """
    train_dataset = MuliTsBatchedWindowDataset(train_data, label=None, device=device, window_size=args.window_size,
                                               stride=1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                              drop_last=True, pin_memory=True)

    test_dataset = MuliTsBatchedWindowDataset(test_data, label=test_label, device=device, window_size=args.window_size,
                                               stride=1)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                              drop_last=True, pin_memory=True)

    print("Begin save train data...")
    model_recons = []
    train_score = []
    raw_x = []
    x_dim = train_data.shape[1]
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(tqdm(train_loader)):
            raw_x.append(x[:,-1,:])
            x = x.to(device)
            # x.shape: batch × window × node
            idx = torch.tensor(np.random.permutation(range(x_dim))).to(device)
            x_rec = model(x, idx, device)
            score = nn.MSELoss(reduction='none')(x_rec[0][0], x_rec[0][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + nn.MSELoss(reduction='none')(x_rec[1][0], x_rec[1][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + \
                    nn.MSELoss(reduction='none')(x_rec[2][0], x_rec[2][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + nn.MSELoss(reduction='none')(x_rec[3][0], x_rec[3][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + \
                    nn.MSELoss(reduction='none')(x_rec[4][0], x_rec[4][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)
            train_score.append(score.detach().cpu().numpy().reshape(-1))
            model_recons.append(x_rec[4][1][:,-1,:].detach().cpu().numpy())

    model_recons = np.concatenate(model_recons, axis=0)
    train_score = np.concatenate(train_score, axis=0)
    actual = np.concatenate(raw_x, axis=0)
    # actual = train_data[args.window_size-1:]
    train_df = pd.DataFrame()
    for i in range(x_dim):
        train_df[f"Recon_{i}"] = model_recons[:, i]
        train_df[f"True_{i}"] = actual[:, i]
    train_df[f"A_Score_Global"] = train_score
    print(f"Saving output to {args.save_path}/<train_output.pkl....")
    train_df.to_pickle(f"{args.save_path}/train_output.pkl")
    print(f"Saving Success!")

    print("Begin save test data....")
    y_pred_score_test_test = []
    y_true_label_test = []
    recons_test = []
    x_raw_test = []
    test_score = []
    x_dim = train_data.shape[1]
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x_raw_test.append(x[:,-1,:])
            x = x.to(device)
            idx = torch.tensor(np.random.permutation(range(x_dim))).to(device)
            x_rec = model(x, idx, device)
            score = nn.MSELoss(reduction='none')(x_rec[0][0], x_rec[0][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + nn.MSELoss(reduction='none')(x_rec[1][0], x_rec[1][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + \
                    nn.MSELoss(reduction='none')(x_rec[2][0], x_rec[2][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + nn.MSELoss(reduction='none')(x_rec[3][0], x_rec[3][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + \
                    nn.MSELoss(reduction='none')(x_rec[4][0], x_rec[4][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)
            # x.shape: batch × window × node
            test_score.append(score.cpu().numpy().reshape(-1))
            recons_test.append(x_rec[4][1][:, -1, :].detach().cpu().numpy())

            y_true_label_test.append(y[:, -1].cpu().numpy().reshape(-1))

    recons_test = np.concatenate(recons_test, axis=0)
    test_score = np.concatenate(test_score, axis=0)
    actual_test = np.concatenate(x_raw_test, axis=0)
    y_true_label_test = np.concatenate(y_true_label_test)

    test_anomaly_scores = np.zeros_like(actual_test)
    df_test = pd.DataFrame()
    for i in range(recons_test.shape[1]):
        df_test[f"Recon_{i}"] = recons_test[:, i]
        df_test[f"True_{i}"] = actual_test[:, i]
        score_i = np.sqrt((recons_test[:, i] - actual_test[:, i]) ** 2)

        test_anomaly_scores[:, i] = score_i
        df_test[f"A_Score_{i}"] = score_i

    test_anomaly_scores = np.mean(test_anomaly_scores, 1)
    test_anomaly_scores = adjust_predicts(test_anomaly_scores, y_true_label_test)

    df_test['A_Score_Global'] = test_score
    df_test['A_True_Global'] = y_true_label_test

    fps, tps, thresholds = _binary_clf_curve(y_true_label_test, test_anomaly_scores)
    # y_pred_label_test = test_anomaly_scores thresholds
    total_pos = tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    fps = fps[sl]
    tps = tps[sl]
    precisions = tps / (fps + tps)
    recalls = tps / total_pos
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_ind = np.nanargmax(f1_scores)
    df_test['Thresh_Global'] = thresholds[f1_ind]
    df_test['A_Pred_Global'] = (test_anomaly_scores > thresholds[f1_ind])+0

    print(f"Saving output to {args.save_path}/<test_output.pkl....")
    df_test.to_pickle(f"{args.save_path}/test_output.pkl")
    print(f"Saving Success!")
