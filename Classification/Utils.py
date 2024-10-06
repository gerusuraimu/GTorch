import os
import pickle
from typing import Union

import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageFile

from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize

import Models
from DataModels import Config


def get_config(model: Union[str, None]) -> Union[Config, None]:
    def open_pickle(file) -> Config:
        with open(file, 'rb') as f:
            conf: Config = pickle.load(f)
        return conf

    def get_pickle(_dir) -> Union[Config, None]:
        conf: Union[Config, None] = None
        config_file: list = [os.path.join(_dir, file) for file in os.listdir(_dir) if file.endswith('.pickle')]
        if config_file and os.path.isfile(config_file[0]):
            conf = get_pickle(config_file[0])
        return conf

    config: Union[Config, None] = None
    if isinstance(model, str):

        # modelがアーキテクチャ名だった場合の処理
        if model in Models.architecture():
            config = Config()
            config.architecture = model
            config.device = config.get_device()

        # modelが訓練結果を保存したディレクトリ名だった場合の処理
        elif os.path.isdir(model):
            config = get_pickle(model)
            config.results = model

        # modelが.pickleファイルだった場合
        elif os.path.isfile(model) and model.endswith('.pickle'):
            config = open_pickle(model)
            config.results = os.path.dirname(model)

        # modelが.pt, .onnx, .engineファイルだった場合
        elif os.path.isfile(model):
            if model.endswith('.pt') or model.endswith('.onnx') or model.endswith('.engine'):
                model_dir = os.path.dirname(model)
                config = get_pickle(model_dir)
                config.results = model_dir

    return config


def get_model(model: str, config: Config):
    pass


def get_transform(config: Config) -> Compose:
    transform = Compose([Resize(config.imgsz),
                         ToTensor(),
                         Normalize(mean=config.mean, std=config.std)])
    return transform


def get_image(transform: Compose, image: Union[str, np.ndarray, ImageFile], config: Config) -> ImageFile:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    return transform(image).unsqueeze(0).to(config.device)
