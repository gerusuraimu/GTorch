import os
from typing import List, Any, Optional
from .DataModels import Config
from . import Utils
from . import Models
from .. import Errors


class GTorchBase:
    def __init__(self, model: Optional[str], train_dir: Optional[str], valid_dir: Optional[str], test_dir: Optional[str]):
        """
        - v0.0 -
        モデルと設定の定義だけ実行する。
        :param model: str  -> 独自データセットで訓練したモデルまたはモデルを含むディレクトリを指定。
                              又はPyTorchが持っている事前学習済みモデルのアーキテクチャを指定。（ResNet50とか）
                              使用できる事前学習済みモデルは"architecture()"で取得できる。
                              => 取得したリスト内の文字列をそのまま引数にすることを想定している。
                      None -> 使用可能なアーキテクチャがわからない時は引数指定できないよね・・・？
                              => インスタンス化の際はarchitecture()以外のメソッドは実行不可になる。
        """

        self.config: Optional[Config] = Utils.get_config(model)

        if self.config is not None:
            self.config.train_dir = train_dir
            self.config.valid_dir = valid_dir
            self.config.test_dir = test_dir
            if self.config.train_dir is not None:
                self.config.classes = len([_dir for _dir in os.listdir(self.config.train_dir) if os.path.isdir(os.path.join(self.config.train_dir, _dir))])

        self.model: Optional[Any] = Utils.get_model(self.config)
        self.is_run: bool = False if model is None else True

    @staticmethod
    def architecture() -> list:
        return Models.architecture()

    def predict(self):
        if not self.is_run:
            self.send_is_run_error()

    def train(self):
        if not self.is_run:
            self.send_is_run_error()

    def benchmark(self):
        if not self.is_run:
            self.send_is_run_error()

    def send_is_run_error(self):
        message = f'\nconfig = {self.config is not None}\nmodel = {self.model is not None}\nis_run = {self.is_run}'
        raise Errors.IsRunException(message)


class GTorch(GTorchBase):
    def __init__(self, model: Optional[str] = None, train_dir: Optional[str] = None, valid_dir: Optional[str] = None, test_dir: Optional[str] = None):
        super().__init__(model, train_dir, valid_dir, test_dir)

    def __call__(self):
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
