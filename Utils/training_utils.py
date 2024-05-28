import torch
from Utils.helper import compute_a_bar, compute_diffusion_noise_coefficient, compute_reverse_diffusion_uncertainty


class DiffusionUtils:

    def __init__(self, beta_start : float, beta_end : float, diffusion_time : int = 100, device = torch.device("cpu")) -> None:
        self.device = device
        self._diffusion_time = diffusion_time
        self.beta_schedule = torch.linspace(beta_start, beta_end, diffusion_time).to(self.device)
        self.A = 1 - self.B
        self.A_bar = compute_a_bar(self.A)
        self.diffusion_noise_coeff = compute_diffusion_noise_coefficient(self.A, self.A_bar)

    def add_noise_to_image(self, image : torch.Tensor, t : torch.Tensor):
        noise_coeff = 1 - self.A_bar[t, None, None, None]
        image_coeff = self.A_bar[t, None, None, None]
        noise = torch.sqrt(noise_coeff)*torch.rand_like(image)
        noisy_img = image*torch.sqrt(image_coeff) + noise
        return noisy_img, noise
    
    def predict_previous_image(self, model, x_t, time, random_init=True):
        t = time.item()
        predicted_noise = model(x_t, time)
        mu = (1/torch.sqrt(self.A[t]))*(x_t - predicted_noise*self.diffusion_noise_coeff[t])
        beta_tilde = self.B[t] if random_init else compute_reverse_diffusion_uncertainty(self.A, self.A_bar)[t]
        new_noise = torch.rand_like(mu)*torch.sqrt(beta_tilde)
        return mu + new_noise
    
    @property
    def B(self):
        return self.beta_schedule

    @property
    def T(self):
        return self._diffusion_time