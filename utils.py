import torch
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim
from skimage.metrics import mean_squared_error as comp_mse


def init_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 42:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=>Saving checkpoint")
    torch.save(state, filename)


def count_percentage(code, mod, epoch, snr, channel_use, tradeoff_h):
    if mod == '4qam' or mod == 'bpsk':
        pass
    else:
        code = code.reshape(-1)
        index = [i for i in range(len(code))]
        random.shuffle(index)
        code = code[index]
        code = code.reshape(-1, 2).cpu()

        if mod == '16qam':
            I_point = torch.tensor([-3, -1, 1, 3])
            order = 16
        elif mod == '64qam':
            I_point = torch.tensor([-7, -5, -3, -1, 1, 3, 5, 7])
            order = 64

        I, Q = torch.meshgrid(I_point, I_point)
        map = torch.cat((I.unsqueeze(-1), Q.unsqueeze(-1)), dim=2).reshape(order, 2)
        per_s = []
        fig = plt.figure(dpi=300)
        ax = Axes3D(fig)
        fig.add_axes(ax)
        for i in range(order):
            temp = torch.sum(torch.abs(code - map[i, :]), dim=1)
            num = code.shape[0] - torch.count_nonzero(temp).item()
            per = num / code.shape[0]
            per_s.append(per)
        per_s = torch.tensor(per_s).cpu()
        height = np.zeros_like(per_s)
        width = depth = 0.3
        surf = ax.bar3d(I.ravel(), Q.ravel(), height, width, depth, per_s, zsort='average')
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
        file_name = './cons_fig/' + '{}_{}_{}_{}'.format(mod, snr, channel_use, tradeoff_h)
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        fig.savefig(file_name + '/{}'.format(epoch))
        plt.close()

        # additional scatter plot
        if mod == '64qam':
            fig = plt.figure(dpi=300)
            for k in range(order):
                plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='b')
            fig.savefig(file_name + '/scatter_{}'.format(epoch))
            plt.close()


def PSNR(tensor_org, tensor_trans):
    total_psnr = 0
    origin = ((tensor_org + 1) / 2).cpu().numpy()
    trans = ((tensor_trans + 1) / 2).cpu().numpy()
    for i in range(np.size(trans, 0)):
        psnr = 0
        for j in range(np.size(trans, 1)):
            psnr_temp = comp_psnr(origin[i, j, :, :], trans[i, j, :, :])
            psnr = psnr + psnr_temp
        psnr /= 3
        total_psnr += psnr
    return total_psnr


def SSIM(tensor_org, tensor_trans):
    total_ssim = 0
    origin = tensor_org.cpu().numpy()
    trans = tensor_trans.cpu().numpy()
    for i in range(np.size(trans, 0)):
        ssim = 0
        for j in range(np.size(trans, 1)):
            ssim_temp = comp_ssim(origin[i, j, :, :], trans[i, j, :, :], data_range=1.0)
            ssim = ssim + ssim_temp
        ssim /= 3
        total_ssim += ssim

    return total_ssim


def MSE(tensor_org, tensor_trans):
    origin = ((tensor_org + 1) / 2).cpu().numpy()
    trans = ((tensor_trans + 1) / 2).cpu().numpy()
    mse = np.mean((origin - trans) ** 2)
    return mse * tensor_org.shape[0]