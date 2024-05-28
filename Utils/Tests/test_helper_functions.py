import unittest
import sys
import torch

# sys.path.append('..')
from Utils.helper import compute_a_bar, compute_diffusion_noise_coefficient


class TestHelperFunctions(unittest.TestCase):

    def test_calculate_a_bar(self):

        A = torch.tensor([1, 0.1, 0.1])
        A_bar = compute_a_bar(A).round(decimals=2)
        expected = torch.tensor([1.0, 0.1, 0.01]).round(decimals=2)
        all_equal = (A_bar == expected).all()
        self.assertTrue(all_equal)

    def test_calculate_diffusion_noise_coefficient(self):

        A = torch.tensor([0.75])
        A_bar = torch.tensor([0.75])

        expected = 0.5
        res = compute_diffusion_noise_coefficient(A, A_bar)
        self.assertEqual(expected, res.item())