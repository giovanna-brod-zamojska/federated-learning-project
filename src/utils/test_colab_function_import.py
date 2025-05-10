import torch

from loguru import logger

def say_hello():
    print("Hello")

def say_hello_with_loguru():
    logger.info("Hello")

def is_cuda_available():
    # colab has already torch installed (?)
    print("Cuda is available:", torch.cuda.is_available())
