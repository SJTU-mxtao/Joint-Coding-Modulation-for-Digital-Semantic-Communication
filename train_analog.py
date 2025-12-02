import os
import torch
from torch import optim, nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from evaluation import EVAL_analog
from utils import save_checkpoint, PSNR


def train_analog(config, net, train_iter, test_iter, device):
    learning_rate = config.lr
    epochs = config.train_iters

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    loss_f1 = nn.CrossEntropyLoss()
    loss_f2 = nn.MSELoss()
    results = {'epoch': [], 'acc': [], 'mse': [], 'psnr': [], 'ssim': [], 'loss': []}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.train_iters+1, T_mult=1, eta_min=1e-6, last_epoch=-1)

    best_acc = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        acc_total_train = 0
        psnr_total_train = 0
        for i, (X, Y) in enumerate(tqdm(train_iter)):
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            _, _, y_class, y_recon = net(X)

            loss_1 = loss_f1(y_class, Y)
            loss_2 = loss_f2(y_recon, X)

            loss = loss_1 + config.tradeoff_lambda * loss_2

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().item())

            # acc & psnr of the train set
            acc = (y_class.data.max(1)[1] == Y.data).float().sum()
            acc_total_train += acc
            psnr = PSNR(X, y_recon.detach())
            psnr_total_train += psnr

        scheduler.step()

        loss = sum(epoch_loss) / len(epoch_loss)
        acc_train = acc_total_train / 50000
        psnr_train = psnr_total_train / 50000

        acc, mse, psnr, ssim = EVAL_analog(net, test_iter, device, config, epoch)
        print('epoch: {:d}, loss: {:.6f}, acc: {:.3f}, mse: {:.6f}, psnr: {:.3f}, ssim: {:.3f}, lr: {:.6f}'.format
              (epoch, loss, acc, mse, psnr, ssim, optimizer.state_dict()['param_groups'][0]['lr']))
        print('train acc: {:.3f}'.format(acc_train))
        print('train psnr: {:.3f}'.format(psnr_train))

        acc_num = acc.detach().cpu().numpy()
        results['epoch'].append(epoch)
        results['loss'].append(loss)
        results['acc'].append(acc_num)
        results['mse'].append(mse)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

        if (epochs - epoch) <= 10 and acc_num > best_acc:
            file_name = config.model_path + '/analog/'
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            model_name = 'CIFAR_SNR{:.3f}_Trans{:d}_analog.pth.tar'.format(
                config.snr_train, config.channel_use, config.mod_method)
            save_checkpoint(net.state_dict(), file_name + model_name)
            best_acc = acc_num

    # in the end save all the results
    data = pd.DataFrame(results)
    file_name = config.result_path + '/analog/'
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    result_name = 'CIFAR_SNR{:.3f}_Trans{:d}_analog.csv'.format(
            config.snr_train, config.channel_use)
    data.to_csv(file_name + result_name, index=False, header=False)