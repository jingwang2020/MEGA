# -*- coding: utf-8 -*-
"""
@Time ： 2021/4/7 10:11
@Auth ： Ssk
@File ：result_csv_combine.py
@IDE ：PyCharm
@Do : Merge the results of various datasets
"""
import os
import pandas as pd


def merge_smd_csv_result_file_usinglstm(file_path, to_combine_folder):

    log_folder = [i for i in os.listdir(file_path) if 'csv' not in i]

    final_result_list = []
    file_list = []
    i=0
    for folder in log_folder:
        f = pd.read_csv(file_path+folder+'/'+'statistics.csv')
        if i == 0:
            header = f[['data_category', 'val_split', 'shuffle_dataset', 'print_model', 'window_size',
                        'hidden_size', 'latent_size', 'top_k', 'embedding_dim', 'gc_depth', 'use_dis', 'a', 'b',
                        'c', 'd', 'e', 'resume', 'load_path', 'save_path', 'save_interval',
                        'save_plot_data', 'num_gpu', 'seed', 'device', 'epochs', 'batch_size',
                        'model_lr', 'critic_iter']]
        i += 1
        file_list.append(f)
    merge = pd.concat(file_list)


    tp_sum = merge['TP'].sum()
    print(tp_sum)
    fp_sum = merge['FP'].sum()
    print(fp_sum)
    fn_sum = merge['FN'].sum()
    print(fn_sum)
    precision = tp_sum/(tp_sum+fp_sum + 0.00001)
    recall = tp_sum/(tp_sum+fn_sum + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    print("f1: {}".format(f1))

    header['PRE'] = precision
    header['RECALL'] = recall
    header['F1'] = f1

    header.to_csv(to_combine_folder + "summary.csv", index=0, sep=',')


def merge_mslsmap_csv_result_file(file_path, to_combine_folder):
    log_csv = [i for i in os.listdir(file_path) if "merge" not in i and "csv" in i]
    file_list = []
    for i in log_csv:
        # Cyclically reading CSV files in the same folder
        f = pd.read_csv(os.path.join(file_path, i))
        file_list.append(f)
    merge = pd.concat(file_list)
    merge.to_csv(to_combine_folder + "result_combine.csv", index=0, sep=',')


if __name__ == '__main__':

    # SMD combine
    smd_result_csv_folder = './MultiWaveExperiment/MultiWaveRecExperiment/parameter_reset_graph_matrix3/new_SMD_adjust_e_other3/e_weight3.5/'
    smd_result_tocombine_folder = './MultiWaveExperiment/MultiWaveRecExperiment/parameter_reset_graph_matrix3/new_SMD_adjust_e_other3/e_weight3.5/'
    #
    merge_smd_csv_result_file_usinglstm(smd_result_csv_folder, smd_result_tocombine_folder)

    # # MSL combine
    # msl_result_csv_folder = '../ICDM_result_save_and_analysis/MSL/using_reg_changebirnn/5.16/'
    # msl_result_tocombine_folder = './ICDM_result_save_and_analysis/MSL/using_reg_changebirnn/'

    # # SMAP combine
    # smap_result_csv_folder = '../ICDM_result_save_and_analysis/SMAP/using_reg_changebirnn/5.16/'
    # smap_result_tocombine_folder = './ICDM_result_save_and_analysis/SMAP/using_reg_changebirnn/'

    # SWAT combine
    # swat_result_csv_folder = '../ICDM_result_save_and_analysis/SWAT/using_reg_changebirnn/5.16/'
    # swat_result_tocombine_folder = './ICDM_result_save_and_analysis/SWAT/using_reg_changebirnn/'

    # merge_mslsmap_csv_result_file(msl_result_csv_folder, msl_result_tocombine_folder)
