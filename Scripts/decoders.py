import torch
import torch.nn as nn


class DecoderBlockv1(nn.Module):

    def __init__(self, 
                 in_chs, 
                 out_chs, 
                 kernel_size=3, 
                 transpose_stride=2, 
                 transpose_padding=1, 
                 out_padding=1,
                 stride=1,
                 padding=1) -> None:
        super(DecoderBlockv1, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*in_chs, out_chs, kernel_size, transpose_stride, transpose_padding, out_padding),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.Conv2d(out_chs, out_chs, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chs),
            nn.ReLU()
        )
    
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.decoder(x)
        return x
    

class DecoderBlockv2(nn.Module):

    def __init__(self, 
                 in_chs, 
                 out_chs,
                 kernel_size=3, 
                 transpose_stride=2, 
                 transpose_padding=1, 
                 out_padding=1,
                 stride=1,
                 padding=1) -> None:
        super(DecoderBlockv2, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*in_chs, out_chs, kernel_size, transpose_stride, transpose_padding, out_padding),
            nn.GroupNorm(out_chs//4, out_chs),
            nn.GELU(),
            nn.Conv2d(out_chs, out_chs, kernel_size, stride, padding),
            nn.GroupNorm(out_chs//4, out_chs),
            nn.GELU(),
            nn.Conv2d(out_chs, out_chs, kernel_size, stride, padding),
            nn.GroupNorm(out_chs//4, out_chs),
            nn.GELU(),
            nn.Conv2d(out_chs, out_chs, kernel_size, stride, padding),
            nn.GroupNorm(out_chs//4, out_chs),
            nn.GELU()
        )
    
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.decoder(x)
        return x