import os
import pickle
from typing import List, Any, Union, Optional

import cv2 as cv
import numpy as np
from PIL import Image, ImageFile

import torch
from numpy.ma.extras import hsplit
from torchvision import models
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

from . import Models
from .DataModels import Config


def get_config(model: Optional[str]) -> Optional[Config]:
    def choose_file(_list: List[str]) -> str:
        for i, file in enumerate(_list):
            print(f'{i}: {file}')

        print('\n使用するファイルの番号を入力してください')
        print(f'例: {_list[0]}を使用する -> "0"と(数字だけ)入力')

        while True:
            index = input('番号を入力: ')

            if index.isdigit():
                index = int(index)

                if 0 <= index < len(_list):
                    ret: str = _list[index]
                    print(f'{ret}を使用します')
                    return ret
                else:
                    print(f'入力値が不正です: {index}')
            else:
                print('数字のみを入力してください')

    def get_model_str(_dir: str) -> Optional[str]:
        model_str: Optional[str] = None
        pt_list: List[str] = [file for file in os.listdir(_dir) if file.endswith('.pt')]
        ox_list: List[str] = [file for file in os.listdir(_dir) if file.endswith('.onnx')]
        en_list: List[str] = [file for file in os.listdir(_dir) if file.endswith('.engine')]
        model_list: List[str] = pt_list + ox_list + en_list

        if len(model_list) == 1:
            model_str: str = model_list[0]
        elif 1 < len(model_list):
            model_str: str = choose_file(model_list)

        return model_str

    def get_pickle(_dir: str, choice: bool = True) -> Optional[Config]:
        conf: Optional[Config] = None
        config_file: List[str] = [os.path.join(_dir, file) for file in os.listdir(_dir) if file.endswith('.pickle')]

        if config_file:
            if len(config_file) == 1:
                file: str = config_file[0]
            else:
                file: str = choose_file(config_file)

            with open(file, 'rb') as pkl:
                conf = pickle.load(pkl)
            conf.results = _dir

            if choice:
                conf.model = get_model_str(_dir)

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
            config.device = config.get_device()

        # modelが.pickleファイルだった場合
        elif os.path.isfile(model) and model.endswith('.pickle'):
            model_dir = os.path.dirname(model)
            with open(model, 'rb') as f:
                config = pickle.load(f)
            config.results = model_dir
            config.model = get_model_str(model_dir)
            config.device = config.get_device()

        # modelが.pt, .onnx, .engineファイルだった場合
        elif os.path.isfile(model):
            if model.endswith('.pt') or model.endswith('.onnx') or model.endswith('.engine'):
                model_dir = os.path.dirname(model)
                config = get_pickle(model_dir, False)
                config.model = os.path.basename(model)
                config.device = config.get_device()

    return config


def get_model(config: Config):
    if config is None:
        return None

    def replace_linear(line: torch.nn.Linear, conf: Config) -> torch.nn.Linear:
        in_features: int = line.in_features
        line = torch.nn.Linear(in_features, conf.classes)
        return line

    def replace_sequential(line: torch.nn.Sequential, conf: Config) -> torch.nn.Sequential:
        in_features: int = line[-1].in_features
        line[-1] = torch.nn.Linear(in_features, conf.classes)
        return line

    model: Any = models.get_model(config.architecture)

    # 出力層のクラス数をconfig.classesに変更する。
    if hasattr(model, 'fc'):
        if isinstance(model.fc, torch.nn.Linear):
            model.fc = replace_linear(model.fc, config)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Linear):
            model.classifier = replace_linear(model.classifier, config)
        elif isinstance(model.classifier[-1], torch.nn.Linear):
            model.classifier[-1] = replace_linear(model.classifier[-1], config)
    elif hasattr(model, 'head'):
        if isinstance(model.head, torch.nn.Linear):
            model.head = replace_linear(model.head, config)
    elif hasattr(model, 'heads'):
        if isinstance(model.heads, torch.nn.Sequential):
            model.heads = replace_sequential(model.heads, config)

    return model


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
