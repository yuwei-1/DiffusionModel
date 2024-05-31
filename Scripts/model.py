import torch
import torch.nn as nn
from Scripts.decoders import DecoderBlockv1, DecoderBlockv2
from Scripts.encoders import EncoderBlockv1, EncoderBlockv2
from Scripts.embedding_blocks import EmbeddingBlockv1, SinusoidalPositionEmbeddingBlock, ClassEmbeddingBlock
    

class UNetv1(nn.Module):

    def __init__(self, im_chs, down_chs=(32, 64, 128), kernel_size=3, stride=1, padding=1):
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
    

class UNetv2(nn.Module):

    def __init__(self, im_chs, num_classes, down_chs=(32, 64, 128), time_embed_dim=10, kernel_size=3, stride=1, padding=1):
        super(UNetv2, self).__init__()

        IMG_SIZE = 16
        latent_image = IMG_SIZE // 4
        up_chs = down_chs[::-1]
        self.T = 100
        
        self.num_classes = num_classes

        self.encoder0 = nn.Sequential(
            nn.Conv2d(im_chs, down_chs[0], kernel_size, stride, padding),
            nn.GroupNorm(down_chs[0]//4, down_chs[0]),
            nn.GELU()
        )

        self.encoder1 = EncoderBlockv2(down_chs[0], down_chs[1])
        self.encoder2 = EncoderBlockv2(down_chs[1], down_chs[2])

        self.to_latent = nn.Sequential(nn.Flatten(), nn.GELU())

        self.compression = nn.Sequential(nn.Linear(down_chs[2]*latent_image**2, down_chs[1]),
                                         nn.GELU(),
                                         nn.Linear(down_chs[1], down_chs[1]),
                                         nn.GELU(),
                                         nn.Linear(down_chs[1], up_chs[0]*latent_image**2),
                                         nn.GELU())
        
        self.decoder0 = nn.Sequential(nn.Unflatten(1, (up_chs[0], latent_image, latent_image)),
                                      nn.Conv2d(up_chs[0], up_chs[0], kernel_size, stride, padding),
                                      nn.GroupNorm(up_chs[0]//4, up_chs[0]),
                                      nn.GELU())

        self.sin_pos_encoding = SinusoidalPositionEmbeddingBlock(time_embed_dim)
        self.emb_t1 = EmbeddingBlockv1(time_embed_dim, up_chs[0])
        self.emb_t2 = EmbeddingBlockv1(time_embed_dim, up_chs[1])
        self.emb_c1 = EmbeddingBlockv1(num_classes, up_chs[0])
        self.emb_c2 = EmbeddingBlockv1(num_classes, up_chs[1])

        self.decoder1 = DecoderBlockv2(up_chs[0], up_chs[1])
        self.decoder2 = DecoderBlockv2(up_chs[1], up_chs[2])

        self.out = nn.Sequential(nn.Conv2d(2*up_chs[2], up_chs[2], kernel_size, stride, padding),
                                              nn.GroupNorm(up_chs[2]//4, up_chs[2]),
                                              nn.GELU(),
                                              nn.Conv2d(up_chs[2], im_chs, kernel_size, stride, padding))

    def forward(self, x, t, c):
        img = self.encoder0(x)
        enc1 = self.encoder1(img)
        enc2 = self.encoder2(enc1)

        x = self.to_latent(enc2)
        x = self.compression(x)

        sin_emb = self.sin_pos_encoding(t)
        emb_t1 = self.emb_t1(sin_emb)
        emb_t2 = self.emb_t2(sin_emb)
        emb_c1 = self.emb_c1(c)
        emb_c2 = self.emb_c2(c)
    
        x = self.decoder0(x)
        x = self.decoder1(x*emb_c1 + emb_t1, enc2)
        x = self.decoder2(x*emb_c2 + emb_t2, enc1)
        x = self.out(torch.cat((img, x), 1))
        return x