from enum import Enum
from torchvision.datasets import FashionMNIST, MNIST
 
class DatasetFactory(Enum):
    FASHIONMNIST = FashionMNIST
    MNIST = MNIST