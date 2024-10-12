from typing import Optional
from torch.nn import Linear
from torch.nn import Conv2d
from torchvision.models import AlexNet
from torchvision.models import alexnet as network
from torchvision.models import AlexNet_Weights as Weights


def alexnet(weights: bool = False, progress: bool = True) -> AlexNet:
    weights: Optional[Weights] = Weights.DEFAULT if weights else None
    model: AlexNet = network(weights=weights, progress=progress)
    return model


def custom(model: AlexNet, config):
    features: int = model.classifier[-1].in_features
    model.classifier[-1]: Linear = Linear(features, config.classes)

    """
    if config.custom_in_gray:
        model.features[0] = Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    """
