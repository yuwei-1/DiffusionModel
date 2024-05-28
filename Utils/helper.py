import torch


def compute_a_bar(A):
    return torch.cumprod(A, 0)

def compute_diffusion_noise_coefficient(A, A_bar):
    return (1-A)/torch.sqrt(1 - A_bar)

def compute_reverse_diffusion_uncertainty(A, A_bar):
    prev_a_bar = A_bar[:-1]
    curr_a_bar = A_bar[1:]
    curr_a = A[1:]
    return (1 - prev_a_bar)*(1 - curr_a)/(1 - curr_a_bar)