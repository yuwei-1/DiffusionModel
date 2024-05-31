import torch.nn as nn
from einops.layers.torch import Rearrange


class EncoderBlockv1(nn.Module):
    
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, padding=1) -> None:
        super(EncoderBlockv1, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.Conv2d(out_chs, out_chs, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)
    

class EncoderBlockv2(nn.Module):
    
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, padding=1) -> None:
        super(EncoderBlockv2, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding),
            nn.GroupNorm(out_chs // 4, out_chs),
            nn.GELU(),
            nn.Conv2d(out_chs, out_chs, kernel_size, stride, padding),
            nn.GroupNorm(out_chs // 4, out_chs),
            nn.GELU(),
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
            nn.Conv2d(4*out_chs, out_chs, kernel_size, stride, padding),
            nn.GroupNorm(out_chs // 4, out_chs),
            nn.GELU()
        )

    def forward(self, x):
        x = self.model(x)
        return x