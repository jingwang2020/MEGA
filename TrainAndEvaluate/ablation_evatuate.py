# -*- coding: utf-8 -*-
# @Time    : 2021/8/12 17:02
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : evatuate.py
# @Software: PyCharm
import torch
from tqdm.std import tqdm
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics._ranking import _binary_clf_curve
import pandas as pd
import os
from .log import set_val_metrics, get_mean_metrics, set_test_metrics
from TrainAndEvaluate.spot import SPOT


def evaluate_metric(model, test_loader, device, x_dim, args, string_output='Start Val Metric evaluating...', epoch=0, test_metric=None, log=None):
    """
        Calculate the evaluation index evaluation on the test set
    """
    print(string_output)
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            # x = torch.unsqueeze(x, dim=1)
            # x = x.transpose(2, 3)
            # idx = torch.tensor(np.random.permutation(range(x_dim))).to(device)
            idx = torch.arange(x_dim).to(device)
            x_rec = model(x, idx, device)

            # score = nn.MSELoss(reduction='none')(x_rec[0][0], x_rec[0][1]).squeeze(dim=1)[:,-1,:].mean(dim=1)+nn.MSELoss(reduction='none')(x_rec[1][0], x_rec[1][1]).squeeze(dim=1)[:,-1,:].mean(dim=1)+\
            #                 nn.MSELoss(reduction='none')(x_rec[2][0], x_rec[2][1]).squeeze(dim=1)[:,-1,:].mean(dim=1)+nn.MSELoss(reduction='none')(x_rec[3][0], x_rec[3][1]).squeeze(dim=1)[:,-1,:].mean(dim=1)
            score = nn.MSELoss(reduction='none')(x_rec, x).squeeze(dim=1)[:,-1,:].mean(dim=1)
            # score = nn.MSELoss(reduction='none')(x_rec[0][0], x_rec[0][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + nn.MSELoss(reduction='none')(x_rec[1][0], x_rec[1][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + nn.MSELoss(reduction='none')(x_rec[2][0], x_rec[2][1]).squeeze(dim=1)[:, -1, :].mean(dim=1)

            # score = nn.MSELoss(reduction='none')(x_rec[0][0], x_rec[0][1]).squeeze(dim=1)[:, -1, :].mean(
            #     dim=1) + nn.MSELoss(reduction='none')(x_rec[1][0], x_rec[1][1]).squeeze(dim=1)[:, -1, :].mean(dim=1) + \
            #         nn.MSELoss(reduction='none')(x_rec[2][0], x_rec[2][1]).squeeze(dim=1)[:, -1, :].mean(
            #             dim=1) + nn.MSELoss(reduction='none')(x_rec[3][0], x_rec[3][1]).squeeze(dim=1)[:, -1, :].mean(
            #     dim=1)

            # score = nn.MSELoss(reduction='none')(x_rec[4][0], x_rec[4][1]).squeeze(dim=1)[:,-1,:].mean(dim=1)

            y_pred.append(score.cpu().numpy().reshape(-1))
            y_true.append(y[:, -1].cpu().numpy().reshape(-1))

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    if args.data_category == 'SMAP' or args.data_category == 'MSL' or args.data_category == 'Swat':
        # bf_performance = calculate_performance(y_true, y_pred, data_category=args.data_category)
        bf_performance = bf_search(y_pred, y_true)
        # train_anomaly_scores = pd.read_pickle(f"{args.save_path}/train_output.pkl")['A_Score_Global']
        # p_eval = pot_eval(train_anomaly_scores, y_pred, y_true, q=1e-3, level=0.5)
        # print("p_eval:")
        # print(p_eval)
        # performance = bf_search(y_pred, y_true, start=0.0, end=0.0001, step_num=100, verbose=False)
        # print("bf_performance")
        # print(bf_performance)
        # print("hello")
    else:
        bf_performance = bf_search(y_pred, y_true)
        # bf_performance = smd_calculate_performance(y_true, y_pred, data_category=args.data_category)
        # train_anomaly_scores = pd.read_pickle(f"{args.save_path}/train_output.pkl")['A_Score_Global']
        # p_eval = pot_eval(train_anomaly_scores, y_pred, y_true, level=0.1)
        # print("p_eval:")
        # print(p_eval)

    test_metric = set_test_metrics(test_metric, bf_performance)
    log.log_test_metrics(test_metric, epoch)

    final_performance = bf_performance

    print('=======================================')
    print('Best performance...')
    print("PR:{0}, REC:{1}, F1:{2}".
          format(final_performance['PR'][0], final_performance['REC'][0], final_performance['F1'][0]))

    print('=======================================')
    return final_performance


def evaluate_loss(model, data_loader, device, x_dim, args, val_loss, string_output='Val loss evaluating...', epoch=0, val_metric=None, log=None, criterion_rec=None):
    """
        Calculate the loss of the validation set and record it, and get the best model on the validation set
    """
    print(string_output)

    data_rec_losses = []
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(tqdm(data_loader)):
            x = x.to(device)
            # x = torch.unsqueeze(x, dim=1)
            # x = x.transpose(2, 3)
            # idx = torch.tensor(np.random.permutation(range(x_dim))).to(device)
            idx = torch.arange(x_dim).to(device)
            output = model(x, idx, device)

            # rec_loss = criterion_rec(output[0][0], output[0][1]) + criterion_rec(output[1][0],output[1][1]) + criterion_rec(output[2][0], output[2][1])
            rec_loss = criterion_rec(output, x)
            # rec_loss = criterion_rec(output[0][0], output[0][1]) + criterion_rec(output[1][0], output[1][1]) + criterion_rec(output[2][0], output[2][1]) + \
            #            criterion_rec(output[3][0], output[3][1])

            # rec_loss = criterion_rec(output[0][0], output[0][1]) + criterion_rec(output[1][0], output[1][1]) + criterion_rec(output[2][0], output[2][1]) + \
            #            criterion_rec(output[3][0], output[3][1])
            # rec_loss = criterion_rec(output[4][0], output[4][1])
            data_rec_losses.append(rec_loss.item())
            val_metric = set_val_metrics(val_metric, rec_loss)

        val_metric = get_mean_metrics(val_metric)

    log.log_val_metrics(val_metric, epoch)
    epoch_val_loss = np.mean(data_rec_losses)

    if epoch_val_loss < val_loss:
        val_loss = epoch_val_loss
        torch.save(model.state_dict(), os.path.join(args.save_path, f'model_best.pth.tar'))

    print("Val_dataset: rec_loss: {0}".format(epoch_val_loss))

    return val_loss


def calculate_performance(y_true, y_prob, verbose=True, data_category="MSL"):
    # print()
    # print(y_true)
    roc_ori = roc_auc_score(y_true, y_prob)
    mAP_ori = average_precision_score(y_true, y_prob)

    print("begin adjust.......................................")
    # y_prob = range_lift_with_delay(y_prob, y_true, delay=delay)

    precisions_ori, recalls_ori, thresholds_ori = precision_recall_curve(y_true, y_prob)
    f1_ori = (2 * precisions_ori * recalls_ori) / (precisions_ori + recalls_ori)
    f1_ind_ori = np.nanargmax(f1_ori)

    y_prob = adjust_predicts(y_prob, y_true, data_category)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_ind = np.nanargmax(f1_scores)

    roc = roc_auc_score(y_true, y_prob)
    mAP = average_precision_score(y_true, y_prob)

    performance = { 'PR_ORI': precisions_ori[f1_ind_ori], 'REC_ORI': [recalls_ori[f1_ind_ori]], 'F1_ORI': [f1_ori[f1_ind_ori]],
                    'PR': [precisions[f1_ind]], 'REC': [recalls[f1_ind]], 'F1': [f1_scores[f1_ind]],
                    'ROC_ORI': [roc_ori], 'mAP_ORI': [mAP_ori], 'ROC': [roc], 'mAP': [mAP]}
    performance = pd.DataFrame(performance)

    if verbose:
        print(performance)
    return performance


def smd_calculate_performance(y_true, y_prob, verbose=True, data_category='machine-1-1'):
    fps_ori, tps_ori, thresholds_ori = _binary_clf_curve(y_true, y_prob)
    total_pos_ori = tps_ori[-1]
    last_ind_ori = tps_ori.searchsorted(tps_ori[-1])
    sl_ori = slice(last_ind_ori, None, -1)
    fps_ori = fps_ori[sl_ori]
    tps_ori = tps_ori[sl_ori]
    precisions_ori = tps_ori / (fps_ori + tps_ori)
    recalls_ori = tps_ori / total_pos_ori
    f1_scores_ori = (2 * precisions_ori * recalls_ori) / (precisions_ori + recalls_ori)
    f1_ind_ori = np.nanargmax(f1_scores_ori)
    fp_ori = fps_ori[f1_ind_ori]
    tp_ori = tps_ori[f1_ind_ori]
    fn_ori = total_pos_ori - tps_ori[f1_ind_ori]

    y_prob_adjust = adjust_predicts(y_prob, y_true, data_category)
    fps, tps, thresholds = _binary_clf_curve(y_true, y_prob_adjust)

    total_pos = tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    fps = fps[sl]
    tps = tps[sl]
    precisions = tps / (fps + tps)
    recalls = tps / total_pos
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_ind = np.nanargmax(f1_scores)
    # print("f1max::::::::::::::::::::::::::::::::::::::::::")
    # print(f1_scores[np.nanargmax(f1_scores)])
    fp = fps[f1_ind]
    tp = tps[f1_ind]
    fn = total_pos - tps[f1_ind]
    # print("metric:::::::::::::::::::::::::::::")
    # pr = tp/(tp+fp)
    # reca = tp/(tp+fn)
    # f1 = (2 * pr * reca) / (pr + reca)
    # print(pr)
    # print(reca)
    # print(f1)
    performance = {'FP_ORI': fp_ori, 'TP_ORI': tp_ori, 'FN_ORI':fn_ori,
                   'PR_ORI': precisions_ori[f1_ind_ori], 'REC_ORI': recalls_ori[f1_ind_ori],'F1_ORI':f1_scores_ori[f1_ind_ori],
                   'FP': fp, 'TP': tp, 'FN': fn,
                   'PR': precisions[f1_ind], 'REC': recalls[f1_ind], 'F1': f1_scores[f1_ind]}
    performance = pd.DataFrame(performance, index=[0])

    if verbose:
        print(performance)

    return performance


def adjust_predicts(score, label, data_category=None):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    new_array = np.copy(score)
    label = np.asarray(label)
    # predict = score > threshold
    # predict = predict.astype(int)
    actual = label == 1
    # print("label::::::::::::::")
    # print(label)
    # print("score::::::::::::::")
    # print(score)
    anomaly_count = 0
    max_score = 0.0
    for i in tqdm(range(len(score))):
        if actual[i]:
            max_score = new_array[i]
            anomaly_count += 1
            for j in range(i - 1, -1, -1):
                if not actual[j]:
                    new_array[j + 1:i + 1] = max_score
                    break
                else:
                    if new_array[j] > max_score:
                        max_score = new_array[j]
    return new_array


def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): boolean list of true anomalies in score
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return dict: pot result dict
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    # pred = adjust_predicts(score, label, pot_th, calc_latency=True)
    pred = adjust_predicts(score, label)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": pot_th,
        }
    else:
        return {
            "threshold": pot_th,
        }


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def bf_search(y_pred, y_true):
    print('Using brute force search for the proper threshold...')

    candidate_values = np.concatenate(
        [y_pred[y_true == 0], np.sort(y_pred[y_true == 1])[:y_pred[y_true == 1].shape[0] // 5]], axis=0)
    candidates = np.linspace(np.min(y_pred[y_true == 0]), np.max(candidate_values), 10000)

    f1s = np.zeros_like(candidates)
    precisions = np.zeros_like(candidates)
    recalls = np.zeros_like(candidates)
    tps = np.zeros_like(candidates)
    tns = np.zeros_like(candidates)
    fps = np.zeros_like(candidates)
    fns = np.zeros_like(candidates)

    y_pred = adjust_predicts(y_pred, y_true)

    def calc_metric(th, num):
        y_res = np.zeros_like(y_pred)
        y_res[y_pred >= th] = 1.0

        p_t = calc_point2point(y_res, y_true)

        f1s[num] = p_t[0]
        precisions[num] = p_t[1]
        recalls[num] = p_t[2]
        tps[num] = p_t[3]
        tns[num] = p_t[4]
        fps[num] = p_t[5]
        fns[num] = p_t[6]

    from threading import Thread

    tasks = []
    for i in tqdm(range(len(candidates))):
        th = Thread(target=calc_metric, args=(candidates[i], i))
        th.start()
        tasks.append(th)

    for th in tasks:
        th.join()

    best_f1_ind = np.argmax(f1s)
    performance = {'F1': f1s[best_f1_ind], 'PR': precisions[best_f1_ind], 'REC': recalls[best_f1_ind],
                   'TP': tps[best_f1_ind], 'TN': tns[best_f1_ind], 'FP': fps[best_f1_ind], 'FN': fns[best_f1_ind]}
    performance = pd.DataFrame(performance, index=[0])

    return performance