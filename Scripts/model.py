import torch
import torch.nn as nn
from decoders import DecoderBlockv1
from encoders import EncoderBlockv1
from embedding_blocks import EmbeddingBlockv1
    

class UNetv1(nn.Module):

    def __init__(self, im_chs, down_chs=(32, 64, 128), latent_dim=64, kernel_size=3, stride=1, padding=1):
        super(UNetv1, self).__init__()

        IMG_SIZE = 16
        latent_image = IMG_SIZE // 4
        up_chs = down_chs[::-1]
        t_dim = 1
        self.T = 100

        self.encoder0 = nn.Sequential(
            nn.Conv2d(im_chs, down_chs[0], kernel_size, stride, padding),
            nn.BatchNorm2d(down_chs[0]),
            nn.ReLU()
        )

        self.encoder1 = EncoderBlockv1(down_chs[0], down_chs[1])
        self.encoder2 = EncoderBlockv1(down_chs[1], down_chs[2])

        self.to_latent = nn.Sequential(nn.Flatten(), nn.ReLU())

        self.compression = nn.Sequential(nn.Linear(down_chs[2]*latent_image**2, down_chs[1]),
                                         nn.ReLU(),
                                         nn.Linear(down_chs[1], down_chs[1]),
                                         nn.ReLU(),
                                         nn.Linear(down_chs[1], up_chs[0]*latent_image**2),
                                         nn.ReLU())
        
        self.decoder0 = nn.Sequential(nn.Unflatten(1, (up_chs[0], latent_image, latent_image)),
                                      nn.Conv2d(up_chs[0], up_chs[0], kernel_size, stride, padding),
                                      nn.BatchNorm2d(up_chs[0]),
                                      nn.ReLU())

        self.emb_t1 = EmbeddingBlockv1(t_dim, up_chs[0])
        self.emb_t2 = EmbeddingBlockv1(t_dim, up_chs[1])

        self.decoder1 = DecoderBlockv1(up_chs[0], up_chs[1])
        self.decoder2 = DecoderBlockv1(up_chs[1], up_chs[2])

        self.out = nn.Sequential(nn.Conv2d(2*up_chs[2], up_chs[2], kernel_size, stride, padding),
                                              nn.BatchNorm2d(up_chs[2]),
                                              nn.ReLU(),
                                              nn.Conv2d(up_chs[2], im_chs, kernel_size, stride, padding))

    def forward(self, x, t):
        img = self.encoder0(x)
        enc1 = self.encoder1(img)
        enc2 = self.encoder2(enc1)

        x = self.to_latent(enc2)
        x = self.compression(x)
        emb_t1 = self.emb_t1(t/self.T)
        emb_t2 = self.emb_t2(t/self.T)
    
        x = self.decoder0(x)
        x = self.decoder1(x + emb_t1, enc2)
        x = self.decoder2(x + emb_t2, enc1)
        x = self.out(torch.cat((img, x), 1))
        return x