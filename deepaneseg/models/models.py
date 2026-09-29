import torch.nn as nn

from deepaneseg.models.building_blocks import (
    number_of_features_per_level,
    create_encoders,
    create_decoders,
    create_conv3d,
    DoubleConv,
)

#######################################  UNET 3D #########################################################


class UNet3D(nn.Module):
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

        super(UNet3D, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the 3D U-Net"

        self.encoders = create_encoders(
            in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, pool_kernel_size
        )

        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order)
        self.final_conv = create_conv3d(f_maps[0], out_channels, kernel=1, padding=0)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        # Decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features[1:]):
            x = decoder(encoder_features, x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x
