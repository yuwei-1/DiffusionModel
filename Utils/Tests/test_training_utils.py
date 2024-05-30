import unittest
import torch
from unittest.mock import patch, MagicMock
from Utils.training_utils import DiffusionUtils


class TestTrainingUtils(unittest.TestCase):
    
    @patch('torch.randn_like')
    def test_forward_diffusion(self, random_tensor):

        start = 0.0001
        end = 0.02
        T = 150
        utils = DiffusionUtils(start, end, diffusion_time=T)

        image = torch.zeros(16,16)
        random_tensor.return_value = torch.rand_like(image)

        noisy_img, noise = utils.forward_diffusion(image, torch.tensor(75))


        def q(x_0, t):
            B = torch.linspace(start, end, T)
            a = 1. - B
            a_bar = torch.cumprod(a, dim=0)
            sqrt_a_bar = torch.sqrt(a_bar)
            sqrt_one_minus_a_bar = torch.sqrt(1 - a_bar)

            t = t.int()
            noise = torch.randn_like(x_0)
            sqrt_a_bar_t = sqrt_a_bar[t, None, None, None]
            sqrt_one_minus_a_bar_t = sqrt_one_minus_a_bar[t, None, None, None]

            x_t = sqrt_a_bar_t * x_0 + sqrt_one_minus_a_bar_t * noise
            return x_t, noise
        
        actual_image, actual_noise = q(image, torch.tensor(75))

        self.assertTrue(torch.allclose(actual_image, noisy_img))
        self.assertTrue(torch.allclose(noise, actual_noise))

    @patch('torch.randn_like')
    def test_reverse_diffusion(self, random_vec):

        start = 0.0001
        end = 0.02
        T = 150
        utils = DiffusionUtils(start, end, diffusion_time=T)

        random_vec.return_value = torch.randn((16,16))
        model = MagicMock(return_value=torch.ones(16, 16))
        t = torch.tensor([[75]])
        x_t = torch.zeros(16, 16)

        def reverse_q(x_t, t, e_t):
            t = torch.squeeze(t[0].int())
            pred_noise_coeff_t = utils.diffusion_noise_coeff[t]
            sqrt_a_inv_t = (1/(torch.sqrt(utils.a)))[t]
            u_t = sqrt_a_inv_t * (x_t - pred_noise_coeff_t * e_t)
            if t == 0:
                return u_t
            else:
                B_t = utils.b[t - 1] # why do they use t - 1 ?
                new_noise = torch.randn_like(x_t)
                return u_t + torch.sqrt(B_t) * new_noise


        pred = utils.reverse_diffusion(model, x_t, t)
        actual = reverse_q(x_t, t, model())

        self.assertTrue(torch.allclose(pred, actual))
