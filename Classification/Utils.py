import os
import re
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
    def open_pickle(file: str) -> Config:
        with open(file, 'rb') as f:
            conf: Config = pickle.load(f)
        return conf

    def get_pickle(_dir: str, choice: bool = False) -> Union[Config, None]:
        conf: Union[Config, None] = None
        config_file: list = [os.path.join(_dir, file) for file in os.listdir(_dir) if file.endswith('.pickle')]

        if config_file:
            if os.path.isfile(config_file[0]):
                conf = open_pickle(config_file[0])
                conf.results = _dir

                if choice:
                    pt_list = [file for file in os.listdir(_dir) if file.endswith('.pt')]
                    ox_list = [file for file in os.listdir(_dir) if file.endswith('.onnx')]
                    en_list = [file for file in os.listdir(_dir) if file.endswith('.engine')]
                    model_list = pt_list + ox_list + en_list

                    if len(model_list) == 1:
                        conf.model = model_list[0]
                    elif 1 < len(model_list):
                        print('--- Choose model file ---')

                        for i, file in enumerate(model_list):
                            print(f'{i}: {file}')

                        print('使用するモデルの番号を入力してください\n')

                        while True:
                            print(f'例: {model_list[0]}を使用する -> "0"と(数字だけ)入力')
                            index = input()
                            if re.match(r'\d+', index):
                                index = int(index)
                                if 0 <= index < len(model_list):
                                    conf.model = model_list[index]
                                    print(f'{conf.model}を使用します')
                                    break
                                else:
                                    print('入力値が不正です')

        return conf

    config: Union[Config, None] = None
    if isinstance(model, str):

        # modelがアーキテクチャ名だった場合
        if model in Models.architecture():
            config = Config()
            config.architecture = model
            config.device = config.get_device()

        # modelが訓練結果を保存したディレクトリ名だった場合
        elif os.path.isdir(model):
            config = get_pickle(model)

        # modelが.pickleファイルだった場合
        elif os.path.isfile(model) and model.endswith('.pickle'):
            model_dir = os.path.dirname(model)
            config = get_pickle(model_dir)

        # modelが.pt, .onnx, .engineファイルだった場合
        elif os.path.isfile(model):
            if model.endswith('.pt') or model.endswith('.onnx') or model.endswith('.engine'):
                model_dir = os.path.dirname(model)
                config = get_pickle(model_dir, False)
                config.model = os.path.basename(model)

    return config


def get_model(config: Config):
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
