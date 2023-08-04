# -*- coding: utf-8 -*-
# @Time    : 2021/8/12 17:02
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : train.py
# @Software: PyCharm
from tqdm.std import tqdm
import torch
import numpy as np
import os
from .evatuate import evaluate_loss
from .log import initialize_train_metrics, get_mean_metrics, set_train_metrics, initialize_val_metrics
from thop import profile

def train(train_loader, val_loader, model, optimizer_rec, scheduler_rec, criterion_rec, device, x_dim, args,
          discriminator=None, optimizer_dis=None, scheduler_dis=None, criterion_dis=None, log=None):
    val_loss = 100000.0
    batch = args.batch_size
    label_real = torch.ones(batch, 1).to(device)
    label_fake = torch.zeros(batch, 1).to(device)
    for epoch in range(args.epochs):
        model.train()
        train_metrics = initialize_train_metrics()
        data_losses = []
        data_dis_losses = []
        data_gen_losses = []
        data_rec_losses = []
        flag = 1
        for i, x in enumerate(tqdm(train_loader, desc='EPOCH: [%d/%d]' % (epoch + 1, args.epochs))):
            x = x.to(device)
            idx = torch.arange(x_dim).to(device)
            if args.use_dis:
                ################### discrimination ###################
                for k in range(args.critic_iter):
                    output = model(x, idx)
                    data_dis_loss = criterion_dis(discriminator(output), label_fake) + criterion_dis(discriminator(x), label_real)
                    discriminator.zero_grad()
                    data_dis_loss.backward()
                    optimizer_dis.step()
                data_dis_losses.append(data_dis_loss.item())
                ###################### data rec ######################
                output = model(x, idx)
                rec_loss = criterion_rec(output, x)
                data_gen_loss = criterion_dis(discriminator(output), label_real)
                data_rec_losses.append(rec_loss.item())
                data_gen_losses.append(data_gen_loss.item())

                loss = data_gen_loss + args.rec_weight * rec_loss
                data_losses.append(loss.item())
                optimizer_rec.zero_grad()
                loss.backward()
                optimizer_rec.step()
                #######################################################
            else:
                ###################### data rec #######################
                output = model(x, idx, device)

                # profile
                if flag == 1:
                    hereflops, params = profile(model, inputs=(x, idx, device))
                    print("flops and params", hereflops, params)
                    flag = 0

                rec_loss = args.a*criterion_rec(output[0][0], output[0][1])+args.b*criterion_rec(output[1][0], output[1][1])+args.c*criterion_rec(output[2][0], output[2][1])\
                           +args.d*criterion_rec(output[3][0], output[3][1])+args.e*criterion_rec(output[4][0], output[4][1])

                data_rec_losses.append(rec_loss.item())
                optimizer_rec.zero_grad()
                rec_loss.backward()
                optimizer_rec.step()
            train_metrics = set_train_metrics(train_metrics, rec_loss)

        train_metrics = get_mean_metrics(train_metrics)
        log.log_train_metrics(train_metrics, epoch)

        print("epoch: {0}, Training_dataset: dis_loss: {1}, rec_loss: {2},  gen_loss: {3}".
              format(epoch + 1, np.mean(data_dis_losses), np.mean(data_rec_losses), np.mean(data_gen_losses)))
        if val_loader is not None:
            val_metric = initialize_val_metrics()
            val_loss = evaluate_loss(model=model, data_loader=val_loader, device=device, x_dim=x_dim, args=args,
                                            val_loss=val_loss, epoch=epoch, val_metric=val_metric, log=log, criterion_rec=criterion_rec)
        if(args.use_dis):
            scheduler_rec.step()
            scheduler_dis.step()
        else:
            scheduler_rec.step()

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'model_epoch{epoch + 1}.pth.tar'))
    torch.save(model.state_dict(), os.path.join(args.save_path, f'final_model.pth.tar'))
