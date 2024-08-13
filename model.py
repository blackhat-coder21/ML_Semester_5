import torch
import torch.nn as nn

class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, activation=nn.ReLU(), norm_layer=None):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.activation = activation
        self.norm_layer = norm_layer

    def forward(self, x):
        conv = self.conv(x)
        mask = self.mask_conv(x)
        if self.norm_layer is not None:
            conv = self.norm_layer(conv)
        gated_conv = torch.sigmoid(mask) * conv
        if self.activation is not None:
            gated_conv = self.activation(gated_conv)
        return gated_conv

class Generator(nn.Module):
    def __init__(self, cnum=48):
        super(Generator, self).__init__()
        self.coarse = nn.Sequential(
            GatedConv(4, cnum, 5, 1, 2),
            GatedConv(cnum, 2 * cnum, 3, 2, 1),
            GatedConv(2 * cnum, 2 * cnum, 3, 1, 1),
            GatedConv(2 * cnum, 4 * cnum, 3, 2, 1),
            GatedConv(4 * cnum, 4 * cnum, 3, 1, 1),
            GatedConv(4 * cnum, 4 * cnum, 3, 1, 1),
            GatedConv(4 * cnum, 4 * cnum, 3, 1, 2, dilation=2),
            GatedConv(4 * cnum, 4 * cnum, 3, 1, 4, dilation=4),
            GatedConv(4 * cnum, 4 * cnum, 3, 1, 8, dilation=8),
            GatedConv(4 * cnum, 4 * cnum, 3, 1, 16, dilation=16),
            GatedConv(4 * cnum, 4 * cnum, 3, 1, 1),
            GatedConv(4 * cnum, 4 * cnum, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            GatedConv(4 * cnum, 2 * cnum, 3, 1, 1),
            GatedConv(2 * cnum, 2 * cnum, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            GatedConv(2 * cnum, cnum, 3, 1, 1),
            GatedConv(cnum, cnum // 2, 3, 1, 1),
            nn.Conv2d(cnum // 2, 3, 3, 1, 1)
        )
        
    def forward(self, x, mask):
        # Concat mask to input image
        x_in = torch.cat([x, mask], dim=1)
        x_coarse = self.coarse(x_in)
        return x_coarse
