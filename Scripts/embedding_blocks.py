import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingBlockv1(nn.Module):

    def __init__(self, input_dim, latent_dim) -> None:
        super(EmbeddingBlockv1, self).__init__()

        self.embedding = nn.Sequential(
                        nn.Linear(input_dim, latent_dim),
                        nn.ReLU(),
                        nn.Linear(latent_dim, latent_dim),
                        nn.ReLU(),
                        nn.Unflatten(1, (latent_dim, 1, 1))
                    )

    def forward(self, x):
        return self.embedding(x)
    

class SinusoidalPositionEmbeddingBlock(nn.Module):

    def __init__(self, time_embed_dim) -> None:
        super(SinusoidalPositionEmbeddingBlock, self).__init__()
        self.time_embed_dim = time_embed_dim
        self.dims = torch.arange(0, self.time_embed_dim, 1, dtype=torch.float32)

    def forward(self, time):
        device = time.device
        batch_dim = time.shape[0]
        ds = self.dims.to(device)
        x = torch.zeros((batch_dim, self.time_embed_dim)).to(device)
        x[:, ::2] = torch.sin(time/(10000**(ds[::2]/self.time_embed_dim)))
        x[:, 1::2] = torch.cos(time/(10000**(ds[::2]/self.time_embed_dim)))
        return x
    

class ClassEmbeddingBlock(nn.Module):

    def __init__(self, num_classes, prob=0.9) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.prob = prob

    def forward(self, c):
        encoded_classes = F.one_hot(c, num_classes=self.num_classes)
        if self.training == False:
            bernoulli_mask = torch.ones_like(encoded_classes, dtype=torch.float32)
        else:
            bernoulli_mask = torch.bernoulli(torch.full_like(encoded_classes, self.prob, dtype=torch.float32))
        return bernoulli_mask*encoded_classes