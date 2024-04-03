from torch import nn
import torch


def normalize(x, power=1):
    power_emp = torch.mean(x ** 2)
    x = (power / power_emp) ** 0.5 * x
    return power_emp, x


class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)
        return x


def awgn(snr, x, device):
    # snr(db)
    n = 1 / (10 ** (snr / 10))
    sqrt_n = n ** 0.5
    noise = torch.randn_like(x) * sqrt_n
    noise = noise.to(device)
    x_hat = x + noise
    return x_hat


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.PReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.prelu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        if config.mod_method == 'bpsk':
            self.layer4 = self.make_layer(ResidualBlock, config.channel_use, 2, stride=2)
        else:
            self.layer4 = self.make_layer(ResidualBlock, config.channel_use * 2, 2, stride=2)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        z0 = self.conv1(x)
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        return z4


class Decoder_Recon(nn.Module):
    def __init__(self, config):
        super(Decoder_Recon, self).__init__()
        self.config = config

        if config.mod_method == 'bpsk':
            input_channel = int(config.channel_use / (4 * 4))
        else:
            input_channel = int(config.channel_use * 2 / (4 * 4))

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 256, 1, 1, 0),
            nn.PReLU())

        self.inchannel = 256

        self.layer1 = nn.Sequential(
            self.make_layer(ResidualBlock, 256, 2, stride=1),
            nn.PReLU())

        self.layer2 = nn.Sequential(
            self.make_layer(ResidualBlock, 256, 2, stride=1),
            nn.PReLU())

        self.DepthToSpace1 = DepthToSpace(4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 128, 1, 1, 0),
            nn.PReLU())

        self.inchannel = 128

        self.layer3 = nn.Sequential(
            self.make_layer(ResidualBlock, 128, 2, stride=1),
            nn.PReLU())

        self.DepthToSpace2 = DepthToSpace(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, 0))

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, z):
        z0 = self.conv1(z.reshape(z.shape[0], -1, 4, 4))
        z1 = self.layer1(z0)
        z2 = self.layer2(z1)
        z3 = self.DepthToSpace1(z2)
        z4 = self.conv2(z3)
        z5 = self.layer3(z4)
        z5 = self.DepthToSpace2(z5)
        z6 = self.conv3(z5)
        return z6


class Decoder_Class(nn.Module):
    def __init__(self, half_width, layer_width):
        super(Decoder_Class, self).__init__()
        self.layer_width = layer_width
        self.Half_width = half_width
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        self.last_fc = nn.Linear(self.layer_width * 4, 10)

    def forward(self, z):
        x1 = self.fc_spinal_layer1(z[:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([z[:, self.Half_width:2 * self.Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([z[:, 0:self.Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([z[:, self.Half_width:2 * self.Half_width], x3], dim=1))
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        y_class = self.last_fc(x)
        return y_class