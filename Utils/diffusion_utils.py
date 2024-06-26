import torch
from Utils.helper_functions import compute_a_bar, compute_diffusion_noise_coefficient, compute_reverse_diffusion_uncertainty


class DiffusionUtils:

    def __init__(self, beta_start : float, beta_end : float, diffusion_time : int = 100, device = torch.device("cpu")) -> None:
        self.device = device
        self._diffusion_time = diffusion_time
        self._beta_schedule = torch.linspace(beta_start, beta_end, diffusion_time).to(self.device)
        self.a = 1 - self.b
        self.a_bar = compute_a_bar(self.a)
        self.diffusion_noise_coeff = compute_diffusion_noise_coefficient(self.a, self.a_bar)
        self.reverse_diffusion_coeff = compute_reverse_diffusion_uncertainty(self.a, self.a_bar)

    def forward_diffusion(self, image : torch.Tensor, t : torch.Tensor):
        noise_coeff = torch.sqrt(1 - self.a_bar)[t, None, None, None]
        image_coeff = torch.sqrt(self.a_bar)[t, None, None, None]
        noise = torch.randn_like(image)
        noisy_img = image*image_coeff + noise*noise_coeff
        return noisy_img, noise
    
    def predict_basic_noise(self, model, x_t, time):
        e_t = model(x_t, time)
        return e_t
    
    def predict_class_noise(self, model, x_t, time, class_vector, w):
        avg_vector = torch.zeros_like(class_vector)
        e_t_cls = model(x_t, time, class_vector)
        e_t_nocls = model(x_t, time, avg_vector)
        e_t = (1 + w) * e_t_cls - w * e_t_nocls
        return e_t
    
    def reverse_diffusion(self, x_t, e_t, time, random_init=True):
        t = time.item()
        norm_coeff = (1/torch.sqrt(self.a[t]))
        mu = norm_coeff*(x_t - e_t*self.diffusion_noise_coeff[t])
        if t == 0:
            return mu
        else:
            beta_tilde = self.b[t] if random_init else self.reverse_diffusion_coeff[t]
            new_noise = torch.randn_like(mu)*torch.sqrt(beta_tilde)
            return mu + new_noise
    
    @property
    def b(self):
        return self._beta_schedule

    @property
    def T(self):
        return self._diffusion_time