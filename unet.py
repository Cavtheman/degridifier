import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import Dropout
from torch.nn import Upsample
from torch.nn import MaxPool2d

import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=[8,16,32,64], dropout=0.5):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.dropout = dropout

        self.pool = MaxPool2d((2,2))

        #Convolutions on the way down
        self.convd1_0 = Conv2d (in_channels, depth[0], (3,3), padding=(1,1), padding_mode="reflect")
        self.convd1_1 = Conv2d (depth[0], depth[0], (3,3), padding=(1,1), padding_mode="reflect")

        self.convd2_0 = Conv2d (depth[0],  depth[1], (3,3), padding=(1,1), padding_mode="reflect")
        self.convd2_1 = Conv2d (depth[1],  depth[1], (3,3), padding=(1,1), padding_mode="reflect")

        self.convd3_0 = Conv2d (depth[1], depth[2], (3,3), padding=(1,1), padding_mode="reflect")
        self.convd3_1 = Conv2d (depth[2], depth[2], (3,3), padding=(1,1), padding_mode="reflect")

        self.convd4_0 = Conv2d (depth[2], depth[2], (3,3), padding=(1,1), padding_mode="reflect")
        self.convd4_1 = Conv2d (depth[2], depth[2], (3,3), padding=(1,1), padding_mode="reflect")

        # Convolutions on the way up
        self.convu1_0 = Conv2d (depth[1], depth[1], (3,3), padding=(1,1), padding_mode="reflect")
        self.convu1_1 = Conv2d (depth[1], depth[1], (3,3), padding=(1,1), padding_mode="reflect")

        self.convu2_0 = Conv2d (depth[0], depth[0], (3,3), padding=(1,1), padding_mode="reflect")
        self.convu2_1 = Conv2d (depth[0], depth[0], (3,3), padding=(1,1), padding_mode="reflect")

        self.up_conv1 = Conv2d (depth[3], depth[1], (1,1), padding=(0,0), padding_mode="reflect")
        self.up_conv2 = Conv2d (depth[2], depth[0], (1,1), padding=(0,0), padding_mode="reflect")
        self.up_conv3 = Conv2d (depth[1], depth[0], (1,1), padding=(0,0), padding_mode="reflect")

        # "Horizontal" convolutions
        self.convu3_0 = Conv2d (depth[0], depth[0], (3,3), padding=(1,1), padding_mode="reflect")
        self.convu3_1 = Conv2d (depth[0], depth[0], (3,3), padding=(1,1), padding_mode="reflect")

        self.convu3_2 = Conv2d (depth[0], 3, (1,1), padding_mode="reflect")

        # Finding the image
        #self.remover_conv1 = Conv2d (in_channels+1, in_channels+1, (3,3), padding=(2,2), padding_mode="reflect", dilation=2)
        self.remover_conv1 = Conv2d (in_channels+1, in_channels+1, (3,3), padding=(1,1), padding_mode="reflect", dilation=1)
        self.remover_conv2 = Conv2d (in_channels+1, out_channels, (3,3), padding=(1,1), padding_mode="reflect")


        self.up_samp1 = Upsample (scale_factor=(2,2), mode='bilinear', align_corners=False)

        self.drop = Dropout (dropout)

        #self.up_conv1 = Upsample(size=(1,depth[3],73,73), mode='nearest')
        #self.up_conv2 = Upsample(size=(1,depth[2],146,146), mode='nearest')
        #self.up_conv3 = Upsample(size=(1,depth[1],292,292), mode='nearest')




    def forward (self, input_data):
        # Downsampling
        conved1_0 = F.elu (self.convd1_0 (input_data))
        conved1_1 = F.elu (self.convd1_1 (conved1_0))
        pooled1 = self.pool (conved1_1)
        #print (pooled1.size ())

        conved2_0 = F.elu (self.convd2_0 (pooled1))
        conved2_1 = F.elu (self.convd2_1 (conved2_0))
        pooled2 = self.pool (conved2_1)
        #print (pooled2.size ())

        conved3_0 = F.elu (self.convd3_0 (pooled2))
        conved3_1 = F.elu (self.convd3_1 (conved3_0))
        dropped1 = self.drop (conved3_1)
        pooled3 = self.pool (dropped1)
        #print (pooled3.size ())

        conved4_0 = F.elu (self.convd4_0 (pooled3))
        conved4_1 = F.elu (self.convd4_1 (conved4_0))
        dropped2 = self.drop (conved4_1)

        #print ("Minimal size:", conved4.size ())

        # Upsampling
        up_sampled1 = self.up_samp1 (dropped2)
        #up_conved1 = torch.cat ((conved3_1, F.elu (self.up_conv1 (up_sampled1))), dim=1)
        up_conved1 = F.elu (self.up_conv1 (torch.cat ((conved3_1, up_sampled1), dim=1)))
        conved5_0 = F.elu (self.convu1_0 (up_conved1))
        conved5_1 = F.elu (self.convu1_1 (conved5_0))

        up_sampled2 = self.up_samp1 (conved5_1)
        #up_conved2 = torch.cat ((conved2_1, F.elu (self.up_conv2 (up_sampled2))), dim=1)
        up_conved2 = F.elu (self.up_conv2 (torch.cat ((conved2_1, up_sampled2), dim=1)))
        conved6_0 = F.elu (self.convu2_0 (up_conved2))
        conved6_1 = F.elu (self.convu2_1 (conved6_0))

        up_sampled3 = self.up_samp1 (conved6_1)
        #up_conved3 = torch.cat ((conved1_1, F.elu (self.up_conv3 (up_sampled3))), dim=1)
        up_conved3 = F.elu (self.up_conv3 (torch.cat ((conved1_1, up_sampled3), dim=1)))
        conved7_0 = F.elu (self.convu3_0 (up_conved3))
        conved7_1 = F.elu (self.convu3_1 (conved7_0))

        # Final horizontal convolution
        #conved8 = F.elu (torch.tanh(self.convu3_2(conved7_1)))
        #conved9 = F.elu (torch.add (input_data,
        #conved8 = torch.add (input_data, torch.tanh(self.convu3_2(conved7_1)))
        #conved8 = F.elu (torch.add (input_data, torch.tanh(self.convu3_2(conved7_1))))

        conved8 = F.elu (torch.add (input_data, torch.tanh(self.convu3_2(conved7_1))))
        clamped = torch.clamp (conved8, 0, 1)

        #conved8 = torch.sigmoid (self.convu3_2 (conved7_1))
        #conved8 = self.convu3_2 (conved7_1)

        #conved8 = torch.sigmoid (torch.add (input_data, torch.tanh(self.convu3_2(conved7_1))))

        # Remover convolutions
        #with_input = torch.cat ((input_data, conved8), dim=1)
        #print ("with_input", with_input.size())
        #conved9 = F.selu (self.remover_conv1 (with_input))
        #print ("first", conved9.size())
        #conved10 = torch.sigmoid (self.remover_conv2 (conved9))
        #print ("second", conved10.size())

        return conved8

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
        }, filename) #model.state_dict(), "temp_best_model.pth")
