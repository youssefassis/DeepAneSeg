"""Experimental models from after the MICCAI 2021 paper: detection coupled with vessel segmentation.

Not used by the published pipeline and not maintained (see deepaneseg/experimental/__init__.py).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from deepaneseg.models.building_blocks import (
    number_of_features_per_level,
    create_encoders,
    create_decoders,
    create_conv3d,
    DoubleConv,
)


#######################################  Proposition 1 #########################################################
class Proposition1(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="cbr",
        num_levels=4,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
    ):

        super(Proposition1, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the 3D U-Net"

        self.encodersDet = create_encoders(
            in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, pool_kernel_size
        )

        self.decodersDet = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order)
        self.final_convDet = create_conv3d(f_maps[0], out_channels, kernel=1, padding=0)

        self.encodersSeg = create_encoders(
            in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, pool_kernel_size
        )

        self.decodersSeg = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order)
        self.final_convSeg = create_conv3d(f_maps[0], out_channels, kernel=1, padding=0)

        self.final_activation = nn.Sigmoid()
        self.attentionblocks = create_attention_blocks(list(reversed(f_maps)))

    def forward(self, x_det, x_seg=None):
        # VESSEL SEGMENTATION
        if x_seg is not None:
            encoders_features_seg = []
            for encoder in self.encodersSeg:
                x_seg = encoder(x_seg)
                encoders_features_seg.insert(0, x_seg)
            seg_features = []
            seg_features.append(x_seg)
            for decoder, encoder_features in zip(self.decodersSeg, encoders_features_seg[1:]):
                x_seg = decoder(encoder_features, x_seg)
                seg_features.append(x_seg)
            x_seg = self.final_convSeg(x_seg)
            x_seg = self.final_activation(x_seg)

        ## ANEURYSM DETECTION
        encoders_features_det = []
        for i, encoder in enumerate(self.encodersDet):
            x_det = encoder(x_det)
            if x_seg is not None and i == len(self.encodersDet) - 1:
                # Attention block
                x_det = self.attentionblocks[0](x_det, seg_features[0])
            encoders_features_det.insert(0, x_det)
        # Decoders
        for i, (decoder, encoder_features) in enumerate(zip(self.decodersDet, encoders_features_det[1:])):
            x_det = decoder(encoder_features, x_det)
            if x_seg is not None:
                # Attention block
                x_det = self.attentionblocks[i + 1](x_det, seg_features[i + 1])
        x_det = self.final_convDet(x_det)
        x_det = self.final_activation(x_det)
        if x_seg is not None:
            return x_det, x_seg
        else:
            return x_det


def create_attention_blocks(f_maps):
    attention_blocks = []
    for i in f_maps:
        attention_blocks.append(Attention(i))
    return nn.ModuleList(attention_blocks)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        if in_channels == 512:
            self.upsampling = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=1, padding=0)
        else:
            self.upsampling = nn.ConvTranspose3d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=True)

    def crop_and_upsample(self, x, d=2):
        dim = x.shape[-1]
        a = (dim - dim // d) // 2
        cropped = x[:, :, a : dim - a, a : dim - a, a : dim - a]
        return self.upsampling(cropped)

    def forward(self, detection, segmentation):
        segmentation = self.crop_and_upsample(segmentation)

        # ReLU activation
        segmentation = self.activation(segmentation)
        # Multiplication
        detection = detection * segmentation
        # conv
        detection = self.conv(detection)
        return detection


class ProjectExciteLayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum(
            [
                squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1),
            ]
        )

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)
        return output_tensor


#######################################  Proposition 2 #########################################################


class Proposition2(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="cbr",
        num_levels=4,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
    ):

        super(Proposition2, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the 3D U-Net"
        ## Detection
        self.encodersDet = create_encoders(
            in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, pool_kernel_size
        )

        self.decodersDet = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order)
        self.DetMultiscaleConvs = create_multiscale_convs(list(reversed(f_maps)))

        ## SEGMENTATION
        self.encodersSeg = create_encoders(
            in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, pool_kernel_size
        )

        self.decodersSeg = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order)
        self.SegMultiscaleConvs = create_multiscale_convs(list(reversed(f_maps)))

        self.final_activation = nn.Sigmoid()
        self.attentionblocks = create_mta_blocks(f_maps[:-1])

    def forward(self, x_det, x_seg=None):
        # VESSEL SEGMENTATION
        if x_seg is not None:
            encoders_features_seg, vessel_predictions = [], []
            for encoder in self.encodersSeg:
                x_seg = encoder(x_seg)
                encoders_features_seg.insert(0, x_seg)
            seg_features = []
            seg_features.append(x_seg)
            vessel_predictions.append(self.final_activation(self.SegMultiscaleConvs[0](x_seg)))
            for i, (decoder, encoder_features) in enumerate(zip(self.decodersSeg, encoders_features_seg[1:])):
                x_seg = decoder(encoder_features, x_seg)
                seg_features.append(x_seg)
                vessel_predictions.append(self.final_activation(self.SegMultiscaleConvs[i + 1](x_seg)))

            x_seg = self.final_activation(vessel_predictions[-1])

        ## ANEURYSM DETECTION
        encoders_features_det, detection_predictions = [], []
        for i, encoder in enumerate(self.encodersDet):
            x_det = encoder(x_det)
            if i == len(self.encodersDet) - 1:  # x_seg is not None and
                detection_predictions.append(self.final_activation(self.DetMultiscaleConvs[0](x_det)))
            encoders_features_det.insert(0, x_det)
        # Decoders
        for i, (decoder, encoder_features) in enumerate(zip(self.decodersDet, encoders_features_det[1:])):
            # Attention
            if x_seg is not None:
                encoder_features = self.attentionblocks[i](
                    encoder_features, detection_predictions[-1], vessel_predictions[i]
                )

            x_det = decoder(encoder_features, x_det)
            detection_predictions.append(self.final_activation(self.DetMultiscaleConvs[i + 1](x_det)))
        x_det = detection_predictions[-1]

        if x_seg is not None:
            return detection_predictions, vessel_predictions
        else:
            return detection_predictions


def create_multiscale_convs(f_maps):
    convs = []
    for m in f_maps:
        convs.append(create_conv3d(m, 1, kernel=1, padding=0))
    return nn.ModuleList(convs)


class MTA(nn.Module):
    def __init__(self, in_channels):
        super(MTA, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=True)

        self.deconv1 = nn.ConvTranspose3d(
            1, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # (1, 1, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose3d(
            1, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # (1, 1, kernel_size=2, stride=2, padding=0)

        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.project_excitation = ProjectExciteLayer(in_channels)

    def forward(self, skip, aneurysm, vessel=None):
        aneurysm = self.deconv1(aneurysm)
        aneurysm = skip * aneurysm
        aneurysm = self.conv1(aneurysm)

        if vessel is not None:
            vessel = self.deconv2(vessel)
            vessel = skip * vessel
            vessel = self.conv2(vessel)
            aneurysm = torch.cat([aneurysm, vessel], dim=1)
            aneurysm = self.conv(aneurysm)

        aneurysm = aneurysm + skip
        return aneurysm


def create_mta_blocks(f_maps):
    attention_blocks = []
    for i in list(reversed(f_maps)):
        attention = MTA(i)
        attention_blocks.append(attention)
    return nn.ModuleList(attention_blocks)


#######################################  Proposition 3 #########################################################


def create_upsampling_blocks(f_maps):
    deconvs = []
    for in_channels in f_maps:
        deconvs.append(
            nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    return nn.ModuleList(deconvs)


class AttentionProp3(nn.Module):
    def __init__(self, in_channels):
        super(AttentionProp3, self).__init__()
        self.upsampling = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # output_padding=1)
        self.conv = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = nn.ReLU()  # inplace=True)

    def forward(self, skip, features):
        features = self.upsampling(features)
        # ReLU activation
        skip = self.activation(skip)

        # Multiplication
        features = features * skip

        features = self.conv(features)
        return features


def create_attention_blocks_prop3(f_maps):
    blocks = []
    for i in f_maps:
        blocks.append(AttentionProp3(i))
    return nn.ModuleList(blocks)


class Proposition3(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        basic_module=DoubleConv,
        f_maps=64,
        layer_order="cbr",
        num_levels=4,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        deep_supervision=True,
    ):

        super(Proposition3, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the 3D U-Net"

        self.encoders = create_encoders(
            in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, pool_kernel_size
        )

        self.decoders = create_decoders(
            f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, upsampling=False
        )
        self.deep_supervision = deep_supervision
        if self.deep_supervision:
            self.DetMultiscaleConvs = create_multiscale_convs(list(reversed(f_maps)))
        else:
            self.final_conv = create_conv3d(f_maps[0], out_channels, kernel=1, padding=0)
        self.final_activation = nn.Sigmoid()
        self.upsampling = create_upsampling_blocks(list(reversed(f_maps[1:])))
        self.attentionBlocks = create_attention_blocks_prop3(list(reversed(f_maps[1:])))

    def forward(self, x):
        # Encoder part
        detection_predictions, encoders_features = [], []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if self.deep_supervision and i == len(self.encoders) - 1:
                detection_predictions.append(self.final_activation(self.DetMultiscaleConvs[0](x)))
            encoders_features.insert(0, x)
        # Decoder part
        for i, (decoder, encoder_features) in enumerate(zip(self.decoders, encoders_features[1:])):
            x = self.upsampling[i](x)
            encoder_features = self.attentionBlocks[i](encoder_features, x)
            x = decoder(encoder_features, x)
            if self.deep_supervision:
                detection_predictions.append(self.final_activation(self.DetMultiscaleConvs[i + 1](x)))
        if not self.deep_supervision:
            detection_predictions = self.final_activation(self.final_conv(x))
        return detection_predictions
