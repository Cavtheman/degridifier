# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, depth=[8,16,32,64]):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, depth[0], 7),
            nn.InstanceNorm2d(depth[0]),
            nn.ELU(),

            # Downsampling
            nn.Conv2d(depth[0], depth[1], 3, stride=2, padding=1),
            nn.InstanceNorm2d(depth[1]),
            nn.ELU(),
            nn.Conv2d(depth[1], depth[2], 3, stride=2, padding=1),
            nn.InstanceNorm2d(depth[2]),
            nn.ELU(),

            # Residual blocks
            ResidualBlock(depth[2]),
            ResidualBlock(depth[2]),
            ResidualBlock(depth[2]),
            ResidualBlock(depth[2]),
            ResidualBlock(depth[2]),
            ResidualBlock(depth[2]),
            ResidualBlock(depth[2]),
            ResidualBlock(depth[2]),
            ResidualBlock(depth[2]),

            # Upsampling
            nn.ConvTranspose2d(depth[2], depth[1], 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(depth[1]),
            nn.ELU(),
            nn.ConvTranspose2d(depth[1], depth[0], 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(depth[0]),
            nn.ELU(),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(depth[0], 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return torch.clamp (torch.add (x, self.main(x)), 0, 1)

    def save(self, filename):
        args_dict = {}
        torch.save({
            "state_dict": self.state_dict(),
            "args_dict": args_dict
        }, filename)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=(1,1), padding_mode="reflect"),
            nn.InstanceNorm2d(in_channels),
            nn.ELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=(1,1), padding_mode="reflect"),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, input_data):
        return torch.add (x, self.res(input_data))
