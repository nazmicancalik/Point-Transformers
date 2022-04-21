import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class ConvolutionBlock(nn.Module):
    def __init__(self,dim,kernel_size,expansion_factor) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.expansion_factor= expansion_factor


        self.layer_norm = nn.LayerNorm(dim)

        self.pointwise_conv_1 = nn.Conv1d(
            in_channels = self.dim,
            out_channels = self.dim * self.expansion_factor,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = True
        )

        self.depthwise_conv = nn.Conv1d(
            in_channels=self.dim,
            out_channels = self.dim,
            kernel_size = self.kernel_size,
            groups=self.dim,
            stride=1,
            padding=(self.kernel_size-1) // 2,
            bias=False
        )

        self.batch_norm = nn.BatchNorm1d(self.dim)
        self.pointwise_conv_2 = nn.Conv1d(
            in_channels= self.dim,
            out_channels = self.dim * self.expansion_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.dropout = nn.Dropout(p=self.dropout)
        self.glu = nn.GLU()

    def forward(self,x):
        x = self.layer_norm(x)
        x = self.pointwise_conv_1(x)
        
        # apply GLU activation
        outputs, gate = x.chunk(2, dim=1)
        x = outputs * gate.sigmoid()
        
        x = self.batch_norm(self.depthwise_conv(x))
        
        # apply swish
        x = x * x.sigmoid()
        
        x = self.dropout(self.pointwise_conv_2(x))
        return x

        