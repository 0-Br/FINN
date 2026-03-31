import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput
from data.feature import NR
from data.dataset import F_BACKGROUND


class UNetConfig():
    """"""

    def __init__(
        self,
        base_channel,
        eps,
        **kwargs
    ):
        """"""
        self.base_channel = base_channel
        self.eps = eps


class UNet(nn.Module):
    def __init__(self, config:UNetConfig):
        super(UNet, self).__init__()

        # Encoder (downsampling with strides for halving spatial dimensions)
        self.enc1 = self.double_conv(len(F_BACKGROUND), config.base_channel, stride=1)
        self.enc2 = self.double_conv(config.base_channel, config.base_channel * 2, stride=1)
        self.enc3 = self.double_conv(config.base_channel * 2, config.base_channel * 4, stride=1)
        self.enc4 = self.double_conv(config.base_channel * 4, config.base_channel * 8, stride=1)

        # here input fully-connected layer
        self.linear = nn.Linear(in_features=NR, out_features=config.base_channel * 16 * 16)

        # Bottleneck
        self.bottleneck = self.double_conv(config.base_channel * 8, config.base_channel * 16, stride=1)

        # Decoder (upsampling)
        rain_encoded_channel = int(config.base_channel * 16 + config.base_channel)
        self.upconv4 = nn.ConvTranspose2d(rain_encoded_channel, config.base_channel * 8, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(config.base_channel * 16, config.base_channel * 8, stride=1)

        self.upconv3 = nn.ConvTranspose2d(config.base_channel * 8, config.base_channel * 4, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(config.base_channel * 8, config.base_channel * 4, stride=1)

        self.upconv2 = nn.ConvTranspose2d(config.base_channel * 4, config.base_channel * 2, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(config.base_channel * 4, config.base_channel * 2, stride=1)

        self.upconv1 = nn.ConvTranspose2d(config.base_channel * 2, config.base_channel, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(config.base_channel * 2, config.base_channel, stride=1)

        # Final output layer
        self.out_conv = nn.Conv2d(config.base_channel, 1, kernel_size=1)

        self.loss_fct = MSELoss()

    def double_conv(self, in_channels, out_channels, stride=1):
        """Double convolution with optional stride: Conv2d -> ReLU -> Conv2d -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, r, y):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))
        # concate rain variables
        r_enc = F.relu(self.linear(r))
        r_enc = r_enc.view(r_enc.shape[0], -1, bottleneck.shape[2], bottleneck.shape[3])
        bottleneck = torch.concat([bottleneck, r_enc], dim=1)

        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))

        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        pred = self.out_conv(dec1)
        loss = self.loss_fct(pred, y)

        return SequenceClassifierOutput(loss=loss, logits=pred)
