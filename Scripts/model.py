import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, padding=1) -> None:
        super(EncoderBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.Conv2d(out_chs, out_chs, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chs),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)
    

class EmbeddingBlock(nn.Module):

    def __init__(self, input_dim, latent_dim) -> None:
        super(EmbeddingBlock, self).__init__()

        self.embedding = nn.Sequential(
                        nn.Linear(input_dim, latent_dim),
                        nn.ReLU(),
                        nn.Linear(latent_dim, latent_dim),
                        nn.ReLU(),
                        nn.Unflatten(1, (latent_dim, 1, 1))
                    )

    def forward(self, x):
        return self.embedding(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, 
                 in_chs, 
                 out_chs, 
                 kernel_size=3, 
                 transpose_stride=2, 
                 transpose_padding=1, 
                 out_padding=1,
                 stride=1,
                 padding=1) -> None:
        super(DecoderBlock, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*in_chs, out_chs, kernel_size, transpose_stride, transpose_padding, out_padding),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(),
            nn.Conv2d(out_chs, out_chs, kernel_size, stride, padding),
            nn.BatchNorm2d(out_chs)
        )
    
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.decoder(x)
        return x
    

class UNet(nn.Module):

    def __init__(self, im_chs, down_chs=(32, 64, 128), latent_dim=64, kernel_size=3, stride=1, padding=1):
        super(UNet, self).__init__()

        IMG_SIZE = 16
        latent_image = IMG_SIZE // 4
        up_chs = down_chs[::-1]
        t_dim = 1
        self.T = 100

        self.img_conv = nn.Sequential(
            nn.Conv2d(im_chs, down_chs[0], kernel_size, stride, padding),
            nn.BatchNorm2d(down_chs[0]),
            nn.ReLU()
        )

        self.encoder1 = EncoderBlock(down_chs[0], down_chs[1])
        self.encoder2 = EncoderBlock(down_chs[1], down_chs[2])

        self.to_latent = nn.Sequential(nn.Flatten(), nn.ReLU())

        self.compression = nn.Sequential(nn.Linear(down_chs[2]*latent_image**2, latent_dim),
                                         nn.ReLU(),
                                         nn.Linear(latent_dim, latent_dim),
                                         nn.ReLU(),
                                         nn.Linear(latent_dim, up_chs[0]*latent_image**2),
                                         nn.ReLU())
        
        self.img_reshape = nn.Unflatten(1, (up_chs[0], latent_image, latent_image))

        self.emb_t1 = EmbeddingBlock(t_dim, up_chs[0])
        self.emb_t2 = EmbeddingBlock(t_dim, up_chs[1])

        self.decoder1 = DecoderBlock(up_chs[0], up_chs[1])
        self.decoder2 = DecoderBlock(up_chs[1], up_chs[2])

        self.back_to_original = nn.Sequential(nn.Conv2d(2*up_chs[2], im_chs, kernel_size, stride, padding), nn.ReLU())


    def forward(self, x, t):
        img = self.img_conv(x)

        enc1 = self.encoder1(img)
        enc2 = self.encoder2(enc1)

        x = self.to_latent(enc2)

        x = self.compression(x)
        x = self.img_reshape(x)

        emb_t1 = self.emb_t1(t/self.T)
        emb_t2 = self.emb_t2(t/self.T)

        x = self.decoder1(x + emb_t1, enc2)
        x = self.decoder2(x + emb_t2, enc1)

        x = self.back_to_original(torch.cat((img, x), 1))

        return x