import os
from typing import Any, Union
from PIL import Image, ImageFile
import numpy as np
import torch
from torchvision.transforms import Compose
from . import Utils
from .DataModels import Config


class Predictor:
    def __init__(self, config: Config, model: Any):
        self.config:    Config  = config
        self.model:     Any     = model
        self.transform: Compose = Utils.get_transform(config)

        weights: str = os.path.join(self.config.results, self.config.model)
        self.model.load_state_dict(torch.load(weights, weights_only=True))
        self.model.to(self.config.device)

    def __call__(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        return self.predict(image)

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        image: ImageFile = Utils.get_image(self.transform, image, self.config.device)
        image: torch.Tensor = image.to(self.config.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
        _, result = torch.max(output.data, 1)
        return result
