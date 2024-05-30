import torch.nn as nn


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