import torch

def say_hello():
    print("Hello")

def is_cuda_available():
    print("Cuda is available:", torch.cuda.is_available())
