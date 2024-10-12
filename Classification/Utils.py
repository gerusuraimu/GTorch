import os
import re
import pickle
from typing import List
from typing import Union
from typing import Optional

import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageFile

import torch
from torchvision.transforms import Resize
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize

import Models
from DataModels import Config


def get_config(model: Optional[str]) -> Optional[Config]:
    def choice_file(_list: List[str]) -> str:
        for i, file in enumerate(_list):
            print(f'{i}: {file}')

        print('\n使用するファイルの番号を入力してください')

        while True:
            print(f'例: {_list[0]}を使用する -> "0"と(数字だけ)入力')
            index = input()
            if re.match(r'\d+', index):
                index = int(index)
                if 0 <= index < len(_list):
                    ret = _list[index]
                    print(f'{ret}を使用します')
                    break
                else:
                    print('入力値が不正です')
        return ret

    def get_pickle(_dir: str, choice: bool = False) -> Optional[Config]:
        conf: Optional[Config] = None
        config_file: List[str] = [os.path.join(_dir, file) for file in os.listdir(_dir) if file.endswith('.pickle')]

        if config_file:
            file: Optional[str] = None
            if len(config_file) == 1:
                file = config_file[0]
            elif 1 < len(config_file):
                file = choice_file(config_file)

            with open(file, 'rb') as pkl:
                conf = pickle.load(pkl)
            conf.results = _dir

            if choice:
                pt_list: List[str] = [file for file in os.listdir(_dir) if file.endswith('.pt')]
                ox_list: List[str] = [file for file in os.listdir(_dir) if file.endswith('.onnx')]
                en_list: List[str] = [file for file in os.listdir(_dir) if file.endswith('.engine')]
                model_list: List[str] = pt_list + ox_list + en_list

                if len(model_list) == 1:
                    conf.model = model_list[0]
                elif 1 < len(model_list):
                    conf.model = choice_file(model_list)

        return conf

    config: Optional[Config] = None
    if isinstance(model, str):

        # modelがアーキテクチャ名だった場合
        if model in set(Models.architecture()):
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


def get_image(transform: Compose, image: Union[str, np.ndarray, ImageFile], device: torch.device) -> ImageFile:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    return transform(image).unsqueeze(0).to(device)
