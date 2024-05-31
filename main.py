import sys
import torch
from Scripts.model import UNetv1, UNetv2
from torch.utils.data import DataLoader
from Utils.training_utils import DiffusionUtils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Utils.helper import load_transformed_FashionMNIST, load_transformed_MNIST


if __name__ == "__main__":

    
    IMG_SIZE = 16
    IMG_CHANNELS = 1
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    start = 0.0001
    end = 0.02
    T = 150
    EPOCHS = 3
    
    dataloader = None
    if len(sys.argv) == 1:
        dataloader = DataLoader(load_transformed_FashionMNIST(IMG_SIZE), batch_size=128, shuffle=True, drop_last=True)
    if len(sys.argv) == 2:
        if sys.argv[1].lower() == "fashionmnist":
            dataloader = DataLoader(load_transformed_FashionMNIST(IMG_SIZE), batch_size=128, shuffle=True, drop_last=True)
        elif sys.argv[1].lower() == "mnist":
            dataloader = DataLoader(load_transformed_MNIST(IMG_SIZE), batch_size=128, shuffle=True, drop_last=True)
    if dataloader is None:
        raise ValueError("Invalid argument for dataset")

    utils = DiffusionUtils(start, end, diffusion_time=T, device=device)

    #model = UNetv1(IMG_CHANNELS, down_chs=(16, 32, 64)).to(device)
    model = UNetv2(IMG_CHANNELS, down_chs=(16, 32, 64)).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    @torch.no_grad()
    def sample_images(n_cols, T):
        intervals = int(T/n_cols)
        fig = plt.figure(figsize=(20,5))
        i=1
        x_t = torch.randn((1, 1, IMG_SIZE, IMG_SIZE)).to(device)
        for t in range(0, T)[::-1]:
            time = torch.tensor([[t]]).to(device)
            x_t = utils.reverse_diffusion(model, x_t, time, random_init=True)
            if t % n_cols == 0:
                ax = fig.add_subplot(1, intervals, i)
                ax.imshow(x_t.squeeze().cpu().numpy())
                ax.axis('off')
                i += 1
        plt.tight_layout()
        plt.show()

    
    i = 1
    total_loss = 0

    for epoch in range(EPOCHS):
        for (x_batch, _) in dataloader:
            optimizer.zero_grad()

            x_batch = x_batch.to(device)
            rand_times = torch.randint(0, T, (x_batch.shape[0],)).to(device)
            noisy_images, noise = utils.forward_diffusion(x_batch, rand_times)
            pred_noise = model(noisy_images, rand_times.unsqueeze(1))
            loss = F.mse_loss(noise, pred_noise)

            loss.backward()
            optimizer.step()
            i += 1
            total_loss += loss.item()

            if i % 100 == 0:
                print("current loss: ", total_loss)
                total_loss = 0
                sample_images(15, T)