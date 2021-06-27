import torch
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Discriminator(nn.Module):
    def __init__(self, in_channels, depth=[8,16,32,64]):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.depth = depth

        self.main = nn.Sequential(
            nn.Conv2d (in_channels, depth[0], 4, stride=2, padding=1),
            nn.ELU (),

            nn.Conv2d (depth[0], depth[1], 4, stride=2, padding=1),
            nn.InstanceNorm2d (depth[1]),
            nn.ELU (),

            nn.Conv2d (depth[1], depth[2], 4, stride=2, padding=1),
            nn.InstanceNorm2d (depth[2]),
            nn.ELU (),

            nn.Conv2d (depth[2], depth[3], 4, padding=1),
            nn.InstanceNorm2d (depth[3]),
            nn.ELU (),

            nn.Conv2d (depth[3], 1, 4, padding=1),
        )

    def forward (self, input_data):
        out = self.main (input_data)
        out = F.avg_pool2d (out, out.size()[2:])
        out = torch.flatten (out, 1)
        return torch.sigmoid(out)

    def save(self, filename):
        args_dict = {
            "in_channels": self.in_channels,
            "depth": self.depth
        }
        torch.save({
            "state_dict": self.state_dict(),
            "args_dict": args_dict
        }, filename)

class Discriminator1(nn.Module):
    def __init__(self, in_channels, depth=[8,16,32,64], dropout=0.5):
        super(Discriminator1, self).__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.dropout = dropout

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

        self.final_conv = nn.Sequential(
            Conv2d (depth[3], 1, (1,1)),
            Flatten(),
            nn.Linear(2500,1)
        )

        #self.linear = nn.Linear (2500, 1)

    def forward (self, input_data):
        layer_1 = self.down_layer_1 (input_data)
        layer_2 = self.down_layer_2 (layer_1)
        layer_3 = self.down_layer_3 (layer_2)
        layer_4 = self.down_layer_4 (layer_3)
        pred = torch.sigmoid (self.final_conv (layer_4))

        #print (final_conv.size())
        #print (torch.flatten (final_conv, 2).size())
        #pred = torch.tanh (self.linear (torch.flatten(final_conv, 2)))

        return pred

    def save(self, filename):
        args_dict = {
            "in_channels": self.in_channels,
            "depth": self.depth,
            "dropout": self.dropout
        }
        torch.save({
            "state_dict": self.state_dict(),
            "args_dict": args_dict
        }, filename)
