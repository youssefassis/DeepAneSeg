import torch
import torch.nn as nn
from torch.nn import functional as F

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

def create_conv3d(inp_feat, out_feat, kernel=3, stride=1, padding=1, bias=True):
    return nn.Conv3d(inp_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=bias)

def create_conv(in_channels, out_channels, kernel_size, order, padding):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rl', 'Non-linearity cannot be the first operation in the layer'
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'c':
            bias = not ('b' in order)
            modules.append(('conv', create_conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels, eps=0.001, momentum=0.99)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.99)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['r', 'l', 'c', 'b']")
    return modules

class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cbr', padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, padding=padding):
            self.add_module(name, module)

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cbr', padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        self.add_module('SingleConv1', SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, padding=padding))
        self.add_module('SingleConv2', SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, padding=padding))

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr', padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         padding=padding)
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x

def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, pool_kernel_size):
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              padding=conv_padding)
        else:
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding)
        encoders.append(encoder)
    return nn.ModuleList(encoders)

# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels, conv_kernel_size=2, scale_factor=(2, 2, 2), basic_module=DoubleConv,
#                  conv_layer_order='cbr', padding=1):
#         super(Decoder, self).__init__()
#         self.upsampling = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
#         self.basic_module = basic_module(in_channels+out_channels, out_channels,
#                                          encoder=False,
#                                          kernel_size=conv_kernel_size,
#                                          order=conv_layer_order,
#                                          padding=padding)

#     def forward(self, encoder_features, x):
#         x = self.upsampling(x)
#         x = torch.cat([encoder_features, x], dim=1)
#         x = self.basic_module(x)
#         return x
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=2, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='cbr', padding=1, upsampling=True):
        super(Decoder, self).__init__()
        self.upsampling = upsampling
        if self.upsampling:
            self.upsampling = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.basic_module = basic_module(in_channels+out_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         padding=padding)

    def forward(self, encoder_features, x):
        if self.upsampling:
            x = self.upsampling(x)
        x = torch.cat([encoder_features, x], dim=1)
        x = self.basic_module(x)
        return x
# def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order):
#     decoders = []
#     reversed_f_maps = list(reversed(f_maps))
#     for i in range(len(reversed_f_maps) - 1):
#         in_feature_num = reversed_f_maps[i]
#         out_feature_num = reversed_f_maps[i + 1]
#         decoder = Decoder(in_feature_num, out_feature_num,
#                           basic_module=basic_module,
#                           conv_layer_order=layer_order,
#                           conv_kernel_size=conv_kernel_size,
#                           padding=conv_padding
#                           )
#         decoders.append(decoder)
#     return nn.ModuleList(decoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, upsampling=True):
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        in_feature_num = reversed_f_maps[i]
        out_feature_num = reversed_f_maps[i + 1]
        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          padding=conv_padding,
                          upsampling = upsampling
                          )
        decoders.append(decoder)
    return nn.ModuleList(decoders)

