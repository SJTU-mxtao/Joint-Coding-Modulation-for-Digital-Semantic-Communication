from torch import nn
import torch
from torch.nn.functional import gumbel_softmax
from torch.nn import init
from modules import Encoder, Decoder_Recon, Decoder_Class, awgn, normalize, ResidualBlock


class AnalogNet(nn.Module):
    def __init__(self, config, device):
        super(AnalogNet, self).__init__()
        self.config = config
        self.device = device

        self.encoder = Encoder(self.config)

        if self.config.mod_method == 'bpsk':
            self.dimension = 1
        else:
            self.dimension = 2

        self.avepool = nn.AvgPool2d(4)

        self.decoder_class = Decoder_Class(int(config.channel_use / 2 * self.dimension), int(config.channel_use / 8 * self.dimension))
        self.decoder_recon = Decoder_Recon(config)

    def forward(self, x):
        z = self.avepool(self.encoder(x)).reshape(x.shape[0], -1)
        power, z = normalize(z)

        if self.config.mode == 'train':
            z_hat = awgn(self.config.snr_train, z, self.device)
        if self.config.mode == 'test':
            z_hat = awgn(self.config.snr_test, z, self.device)

        y_class = self.decoder_class(z_hat)
        y_recon = self.decoder_recon(z_hat)
        return z, z_hat, y_class, y_recon