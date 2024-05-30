import torch.nn as nn


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