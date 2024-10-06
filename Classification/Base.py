from typing import Any
from typing import Union
from DataModels import Config
import Utils
import Models
import Errors


class GTorchBase:
    def __init__(self, model: Union[str, None]):
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

        self.config: Union[Config, None] = Utils.get_config(model)
        self.model: Union[Any, None] = Utils.get_model(model, self.config)
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
    def __init__(self, model: Union[str, None] = None):
        super().__init__(model)

    def __call__(self):
        return self.predict()

    def __repr__(self) -> str:
        scripts  = '= 関数一覧 =\n'
        scripts += '  predict(image: Union[str, numpy.ndarray, PIL.ImageFile]) ... 推論を行う関数\n'
        scripts += '     -> 引数"image"は画像ファイルのパス、OpenCV、Pillowのいずれかの形式を想定しています。\n'
        scripts += '  train() ... 訓練を行う関数\n'
        scripts += '     -> "config"の設定に問題がなければ訓練を実行します。\n'
        scripts += '  benchmark() ... ベンチマークを行う関数。\n'
        scripts += '     -> "config"の"test_dir"に設定されたパスにあるテスト用データで精度と速度のテストを行い、結果を表示します。\n'
        scripts += '  architecture() ... 使用できるモデル一覧を取得\n'
        scripts += '     -> この関数で取得できる文字列をそのままインスタンス化時の引数として使用してください。(ResNet18とか)\n'
        return scripts


mod = GTorch('ResNet18')
print(mod.config)
