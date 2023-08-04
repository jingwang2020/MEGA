# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 16:03
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : Ablation.py
# @Software: PyCharm
import torch
import pandas as pd
from DataProcess.dataread import multi_get_data, swat_multi_get_data
from DataProcess.dataset import MuliTsBatchedWindowDataset, create_data_loaders
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import os
from TrainAndEvaluate.ablation_train import train
from TrainAndEvaluate.ablation_evatuate import evaluate_metric
from TrainAndEvaluate.log import Log
from AstMad.ASTMAD import ASTMAD, DataDiscriminator
# from AblationModel.MultWaveGCUNet2DecomposeLayerWithGCN import MultWaveGCUNet2DecomposeLayerWithGCN
from AblationModel.MultWaveGCUNet2DecomposeLayerWithOutGCN import MultWaveGCUNet2DecomposeLayerWithOutGCN
# from AblationModel.MultWaveGCUNet1DecomposeLayerWithGCN import MultWaveGCUNet1DecomposeLayerWithGCN
# from AblationModel.MultWaveGCUNet1DecomposeLayerWithOutGCN import MultWaveGCUNet1DecomposeLayerWithOutGCN
from AblationModel.MultWaveGCUNetWithoutDWT import MultWaveGCUNetWithoutDWT

# from AstMad.MultWaveGCUNet import MultWaveGCUNet
from args import parse_args, setup_seed
from TrainAndEvaluate.log import initialize_test_metrics
from TrainAndEvaluate.save_plot_data import save_train_test


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("Save path not exists, save folder make success")
        print("----------------------------------------------------------------")

    log = Log(args.save_path, 'train_and_eval.log', 'w')

    if args.seed is not None:
        setup_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.num_gpu)
    if args.num_gpu >= 0 and torch.cuda.is_available():
        print("Use Gpu:", os.environ["CUDA_VISIBLE_DEVICES"])
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    # Prepare dataset
    if args.data_category == 'Swat':
        (x_train, _), (x_test, y_test) = swat_multi_get_data(args.data_category)
    else:
        (x_train, _), (x_test, y_test) = multi_get_data(args.data_category)

    x_dim = x_train.shape[1]
    print("x dimension: {}".format(x_dim))
    print("-------------------------------------------------------")
    if args.top_k > x_dim:
        raise ValueError('Input top k greater than the x dimension')

    train_dataset = MuliTsBatchedWindowDataset(x_train, label=None, device=device, window_size=args.window_size,
                                               stride=1)
    test_dataset = MuliTsBatchedWindowDataset(x_test, label=y_test, device=device, window_size=args.window_size,
                                              stride=1)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, args.batch_size, args.val_split, args.shuffle_dataset, test_dataset=test_dataset
    )

    model = MultWaveGCUNetWithoutDWT(input_channel=x_dim, embedding_dim=args.embedding_dim, top_k=args.top_k,
                           input_node_dim=1, graph_alpha=3, device=device, gc_depth=args.gc_depth,
                           batch_size=args.batch_size)

    model = model.to(device)
    optimizer_rec = optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=2.5 * 1e-5)
    scheduler_rec = torch.optim.lr_scheduler.StepLR(optimizer_rec, step_size=20, gamma=0.5)
    if args.print_model:
        print(model)

    rec_criterion = nn.MSELoss().to(device)

    if args.use_dis:
        discriminator = DataDiscriminator(args.window_size, x_dim, args.hidden_size)
        discriminator = discriminator.to(device)
        optimizer_dis = optim.Adam(discriminator.parameters(), lr=args.dis_lr)
        scheduler_dis = torch.optim.lr_scheduler.StepLR(optimizer_dis, step_size=50, gamma=0.75)
        dis_criterion = nn.BCEWithLogitsLoss().to(device)
        if args.print_model:
            print(discriminator)
    else:
        discriminator = None
        optimizer_dis = None
        scheduler_dis = None
        dis_criterion = None

    train(train_loader, val_loader, model, optimizer_rec, scheduler_rec, rec_criterion,
          device, x_dim, args, discriminator, optimizer_dis, scheduler_dis, dis_criterion, log)
    torch.save(model.state_dict(), os.path.join(args.save_path, f'model_final.pth.tar'))

    model.load_state_dict(torch.load(os.path.join(args.save_path, f'model_best.pth.tar'), map_location=device))

    test_metric = initialize_test_metrics()

    final_performance = evaluate_metric(model=model, test_loader=test_loader, device=device, x_dim=x_dim, args=args,
                                        string_output='Best evaluating...', epoch=args.epochs, test_metric=test_metric,
                                        log=log)

    model.load_state_dict(torch.load(os.path.join(args.save_path, f'final_model.pth.tar'), map_location=device))

    save_path = args.save_path
    args = vars(args)

    if args['data_category'] == 'SMAP' or args['data_category'] == 'MSL' or args['data_category'] == 'Swat':
        args['PR'] = final_performance['PR'][0]
        args['REC'] = final_performance['REC'][0]
        args['F1'] = final_performance['F1'][0]

    else:
        args['PR'] = final_performance['PR'][0]
        args['REC'] = final_performance['REC'][0]
        args['F1'] = final_performance['F1'][0]
        args['TP'] = final_performance['TP'][0]
        args['FP'] = final_performance['FP'][0]
        args['FN'] = final_performance['FN'][0]

    df = pd.DataFrame(args, index=[0])
    if args['data_category'] == 'SMAP':
        df.to_csv(os.path.join(save_path, f'SMAP_statistics.csv'), index=False)
    elif args['data_category'] == 'MSL':
        df.to_csv(os.path.join(save_path, f'MSL_statistics.csv'), index=False)
    elif args['data_category'] == 'Swat':
        df.to_csv(os.path.join(save_path, f'Swat_statistics.csv'), index=False)
    else:
        df.to_csv(os.path.join(save_path, f'statistics.csv'), index=False)