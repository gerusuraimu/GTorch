import os
from typing import List, Any, Union, Optional
import torch
import numpy as np
from PIL import Image
from .DataModels import Config
from . import Utils
from . import ModelList
from . import Trainer
from . import Predictor
from .. import Errors


class GTorchBase:
    def __init__(self, model: Optional[str], train_dir: str, valid_dir: str, test_dir: str):
        """
        モデルと設定の定義だけ実行する。
        :param model: str  -> 独自データセットで訓練したモデルもしくはモデルを含むディレクトリを指定。
                              又はPyTorchが持っている事前学習済みモデルのアーキテクチャを指定。（ResNet50とか）
                              使用できる事前学習済みモデルは"architecture()"で取得できる。
                              => 取得したリスト内の文字列をそのまま引数にすることを想定している。
                      None -> 使用可能なアーキテクチャがわからない時は引数指定できないよね・・・？
                              => インスタンス化の際はarchitecture()以外のメソッドは実行不可になる。
        """

        self.config: Optional[Config] = Utils.get_config(model)

        self.config.train_dir = train_dir
        self.config.valid_dir = valid_dir
        self.config.test_dir = test_dir

        if self.config is not None and self.config.classes is None:
            self.config.classes = len([_dir for _dir in os.listdir(self.config.train_dir) if os.path.isdir(os.path.join(self.config.train_dir, _dir))])

        self.model: Optional[Any] = Utils.get_model(self.config)
        self.is_run: bool = False if model is None else True

    @staticmethod
    def architecture() -> list:
        return ModelList.architecture()

    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        if not self.is_run:
            self.send_is_run_error()

        predictor = Predictor.Predictor(self.config, self.model)
        return predictor(image)

    def train(self):
        if not self.is_run:
            self.send_is_run_error()

        trainer = Trainer.Trainer(self.config, self.model)
        trainer()

    def benchmark(self):
        if not self.is_run:
            self.send_is_run_error()

    def send_is_run_error(self):
        message = f'\nconfig = {self.config is not None}\nmodel = {self.model is not None}\nis_run = {self.is_run}'
        raise Errors.IsRunException(message)


class GTorch(GTorchBase):
    def __init__(self,
                 model:     Optional[str] = None,
                 train_dir: str           = 'dataset/train',
                 valid_dir: str           = 'dataset/valid',
                 test_dir:  str           = 'dataset/test'):

        super().__init__(model, train_dir, valid_dir, test_dir)

    def __call__(self, image: Union[str, np.ndarray, Image.Image]):
        return self.predict()

    def __repr__(self) -> str:
        lines: List[str] = [
            '= 関数一覧 =',
            '  predict(image: Union[str, numpy.ndarray, PIL.ImageFile]) ... 推論を行う関数',
            '     -> 引数"image"は画像ファイルのパス、OpenCV、Pillowのいずれかの形式を想定しています。',
            '  train() ... 訓練を行う関数',
            '     -> "config"の設定に問題がなければ訓練を実行します。',
            '  benchmark() ... ベンチマークを行う関数。',
            '     -> "config"の"test_dir"に設定されたパスにあるテスト用データで精度と速度のテストを行い、結果を表示します。',
            '  architecture() ... 使用できるモデル一覧を取得',
            '     -> この関数で取得できる文字列をそのままインスタンス化時の引数として使用してください。(ResNet18とか)'
        ]
        return '\n'.join(lines)
