import torch
from torchvision import transforms
from Utils.dataset_factory import DatasetFactory
from torch.utils.data import ConcatDataset


def compute_a_bar(a):
    return torch.cumprod(a, 0)

def compute_diffusion_noise_coefficient(a, a_bar):
    return (1-a)/torch.sqrt(1 - a_bar)

def compute_reverse_diffusion_uncertainty(a, a_bar):
    prev_a_bar = a_bar[:-1]
    curr_a_bar = a_bar[1:]
    curr_a = a[1:]
    return (1 - prev_a_bar)*(1 - curr_a)/(1 - curr_a_bar)

def load_dataset(dataset : str, data_transform):
    dataset = dataset.upper()
    path = "./Data/"
    kwargs = {"download" : True, "transform" : data_transform}
    return ConcatDataset([DatasetFactory[dataset].value(path, train = True, **kwargs), 
                          DatasetFactory[dataset].value(path, train = False, **kwargs)])

def load_transformed_FashionMNIST(IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
    data_transform = transforms.Compose(data_transforms)
    return load_dataset("FashionMNIST", data_transform)

def load_transformed_MNIST(IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ]
    data_transform = transforms.Compose(data_transforms)
    return load_dataset("MNIST", data_transform)