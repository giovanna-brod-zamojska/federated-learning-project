import torch

class CentralizedBaselineTrainer:

    def __init__(self):
        # Load DINO ViT-S/16 model
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        print(f"Untrained centralized baseline ViT-S/16 model loaded {self.model}")

        