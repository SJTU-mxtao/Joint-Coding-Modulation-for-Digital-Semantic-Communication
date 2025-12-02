import torchvision
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from network import JCM
from analog_network import AnalogNet
from train import train
from train_analog import train_analog
from evaluation import EVAL
from utils import init_seeds
import os
import argparse


def mischandler(config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


def main(config):
    # initialize random seed
    init_seeds()

    # prepare training & test data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transform_train,
        download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        transform=transform_test,
        download=True
    )

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = JCM(config, device).to(device)

    if config.pretrain_analog:
        pretrain_net = AnalogNet(config, device).to(device)

        print("Pretraining the analog network.")
        train_analog(config, pretrain_net, train_loader, test_loader, device)

    else:
        if config.mode == 'train':
            print("Loading the pretrained analog network...")

            if config.mod_method == 'bpsk':
                model_name = '/pretrained_analog/1d/CIFAR_analog_SNR{:.3f}_Trans{:d}.pth.tar'.format(
                             config.snr_train, config.channel_use)
            else:
                model_name = '/pretrained_analog/2d/CIFAR_analog_SNR{:.3f}_Trans{:d}.pth.tar'.format(
                             config.snr_train, config.channel_use)
                # the output dimension of BPSK is 1/2 of that of M-QAM

            pretrained_dict = torch.load('./checkpoints' + model_name, map_location=torch.device('cpu'))
            model_dict = net.state_dict()
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict, strict=False)
            print('Successfully load the pretrained analog model!')

            print("Training with the modulation scheme {}.".format(config.mod_method))
            train(config, net, train_loader, test_loader, device)

        elif config.mode == 'test':
            print("Start Testing.")

            if config.load_checkpoint:
                model_name = '/{}/'.format(config.mod_method) + \
                             'CIFAR_SNR{:.3f}_Trans{:d}_{}.pth.tar'.format(
                                 config.snr_train, config.channel_use, config.mod_method)
                net.load_state_dict(torch.load('./checkpoints' + model_name, map_location=torch.device('cpu')))

            acc, mse, psnr, ssim = EVAL(net, test_loader, device, config)
            print('acc: {:.3f}, mse: {:3f}, psnr: {:.3f}, ssmi: {:.3f}'.format(acc, mse, psnr, ssim))

        else:
            print("Wrong mode input!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--channel_use', type=int, default=128)
    """Available modulation methods:"""
    """bpsk, 4qam, 16qam, 64qam"""
    parser.add_argument('--mod_method', type=str, default='4qam')
    parser.add_argument('--load_checkpoint', type=int, default=1)
    parser.add_argument('--pretrain_analog', type=int, default=1)

    # training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--snr_train', type=float, default=12)
    parser.add_argument('--snr_test', type=float, default=12)
    """The tradeoff hyperparameter lambda between two tasks"""
    parser.add_argument('--tradeoff_lambda', type=float, default=200)

    # misc
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--dataset_path', type=str, default='./dataset')

    config = parser.parse_args()

    mischandler(config)
    main(config)
