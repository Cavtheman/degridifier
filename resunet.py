import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import Dropout
from torch.nn import Upsample
from torch.nn import MaxPool2d

import torch.nn.functional as F

class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=[4,8,16,32,64], dropout=0.5):
        super(ResUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.dropout = dropout

        self.pool = MaxPool2d((2,2))

        self.up_samp1 = Upsample (scale_factor=(2,2), mode='bilinear', align_corners=False)

        self.drop = Dropout (dropout)

        self.res_1 = Conv2d (in_channels, depth[0], (1,1))
        self.res_2 = nn.Sequential (
            Conv2d (depth[0], depth[1], (1,1), stride=2),
            nn.BatchNorm2d (depth[1])
        )
        self.res_3 = nn.Sequential (
            Conv2d (depth[1], depth[2], (1,1), stride=2),
            nn.BatchNorm2d (depth[2])
        )
        self.res_4 = nn.Sequential (
            Conv2d (depth[2], depth[3], (1,1), stride=2),
            nn.BatchNorm2d (depth[3])
        )

        self.res_5 = nn.Sequential (
            Conv2d (depth[3]+depth[2], depth[2], (1,1)),
            nn.BatchNorm2d (depth[2])
        )
        self.res_6 = nn.Sequential (
            Conv2d (depth[2]+depth[1], depth[1], (1,1)),
            nn.BatchNorm2d (depth[1])
        )
        self.res_7 = nn.Sequential (
            Conv2d (depth[1]+depth[0], depth[0], (1,1)),
            nn.BatchNorm2d (depth[0])
        )

        self.down_layer_1 = nn.Sequential(
            Conv2d (in_channels, depth[0], (3,3), padding=(1,1), padding_mode="reflect"),
            nn.BatchNorm2d (depth[0]),
            nn.ELU (),
            Conv2d (depth[0], depth[0], (3,3), padding=(1,1), padding_mode="reflect")
        )
        self.down_layer_2 = nn.Sequential(
            nn.BatchNorm2d (depth[0]),
            nn.ELU (),
            Conv2d (depth[0], depth[1], (3,3), stride=2, padding=(1,1), padding_mode="reflect"),
            nn.BatchNorm2d (depth[1]),
            nn.ELU (),
            Conv2d (depth[1], depth[1], (3,3), padding=(1,1), padding_mode="reflect")
        )
        self.down_layer_3 = nn.Sequential(
            nn.BatchNorm2d (depth[1]),
            nn.ELU (),
            Conv2d (depth[1], depth[2], (3,3), stride=2, padding=(1,1), padding_mode="reflect"),
            nn.BatchNorm2d (depth[2]),
            nn.ELU (),
            Conv2d (depth[2], depth[2], (3,3), padding=(1,1), padding_mode="reflect")
        )
        self.down_layer_4 = nn.Sequential(
            nn.BatchNorm2d (depth[2]),
            nn.ELU (),
            Conv2d (depth[2], depth[3], (3,3), stride=2, padding=(1,1), padding_mode="reflect"),
            nn.BatchNorm2d (depth[3]),
            nn.ELU (),
            Conv2d (depth[3], depth[3], (3,3), padding=(1,1), padding_mode="reflect")
        )
        self.up_layer_1 = nn.Sequential(
            nn.BatchNorm2d (depth[3]+depth[2]),
            nn.ELU (),
            Conv2d (depth[3]+depth[2], depth[2], (3,3), padding=(1,1), padding_mode="reflect"),
            nn.BatchNorm2d (depth[2]),
            nn.ELU (),
            Conv2d (depth[2], depth[2], (3,3), padding=(1,1), padding_mode="reflect")
        )
        self.up_layer_2 = nn.Sequential(
            nn.BatchNorm2d (depth[2]+depth[1]),
            nn.ELU (),
            Conv2d (depth[2]+depth[1], depth[1], (3,3), padding=(1,1), padding_mode="reflect"),
            nn.BatchNorm2d (depth[1]),
            nn.ELU (),
            Conv2d (depth[1], depth[1], (3,3), padding=(1,1), padding_mode="reflect")
        )
        self.up_layer_3 = nn.Sequential(
            nn.BatchNorm2d (depth[1]+depth[0]),
            nn.ELU (),
            Conv2d (depth[1]+depth[0], depth[0], (3,3), padding=(1,1), padding_mode="reflect"),
            nn.BatchNorm2d (depth[0]),
            nn.ELU (),
            Conv2d (depth[0], depth[0], (3,3), padding=(1,1), padding_mode="reflect")
        )
        self.final_conv = Conv2d (depth[0], out_channels, (1,1))


    def forward (self, input_data):
        # Encoding
        conv_layer_1 = torch.add (self.res_1 (input_data), self.down_layer_1(input_data))
        conv_layer_2 = torch.add (self.res_2 (conv_layer_1), self.down_layer_2(conv_layer_1))
        conv_layer_3 = torch.add (self.res_3 (conv_layer_2), self.down_layer_3(conv_layer_2))

        # "Bridge"
        conv_layer_4 = torch.add (self.res_4 (conv_layer_3), self.down_layer_4(conv_layer_3))

        # Decoding
        up_sampled_1 = torch.cat ((conv_layer_3, self.up_samp1 (conv_layer_4)), dim=1)
        up_layer_1 = torch.add (self.res_5 (up_sampled_1), self.up_layer_1 (up_sampled_1))

        up_sampled_2 = torch.cat ((conv_layer_2, self.up_samp1 (up_layer_1)), dim=1)
        up_layer_2 = torch.add (self.res_6 (up_sampled_2), self.up_layer_2 (up_sampled_2))

        up_sampled_3 = torch.cat ((conv_layer_1, self.up_samp1 (up_layer_2)), dim=1)
        up_layer_3 = torch.add (self.res_7 (up_sampled_3), self.up_layer_3 (up_sampled_3))

        #final_conv = torch.sigmoid (self.final_conv (up_layer_3))
        final_conv = torch.clamp (torch.add (input_data, self.final_conv (up_layer_3)), 0, 1)

        #final_conv = torch.clamp (self.final_conv (up_layer_3), 0, 1)

        #final_conv = self.final_conv (up_layer_3)
        return final_conv

    def save(self, filename):
        args_dict = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "depth": self.depth,
            "dropout": self.dropout
        }
        torch.save({
            "state_dict": self.state_dict(),
            "args_dict": args_dict
        }, filename)
