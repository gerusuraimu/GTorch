import os
from typing import Union, Any
from dataclasses import dataclass
import torch


@dataclass
class ConfigBase:
    architecture:  str                = None                   # 使用アーキテクチャ名
    results:       str                = None                   # 訓練したモデルを保存しているディレクトリ
    model:         str                = None                   # 保存される（読み込んだ）モデルのファイル名
    imgsz:         Union[int, tuple]  = (320, 320)             # 入力画像のサイズ ... (H, W)
    classes:       int                = None                   # 分類クラス数

    epoch:         int                = 100                    # 学習エポック数
    batch:         int                = 64                     # 学習バッチサイズ
    worker:        int                = 2                      # 学習時ワーカー数
    lr:            float              = 0.0001                 # 学習率
    weights:       bool               = True                   # PyTorchの事前学習済み重みを使用するかどうか
    early_stop:    int                = 50                     # 学習中に精度が向上しなかった場合に早期学習終了を判断するエポック数

    device:        torch.device       = None                   # 学習に使用するデバイス(CUDA, MPS, CPU)
    pin_memory:    bool               = True                   # ピンメモリーを使用するかどうか
    mean:          tuple              = (0.485, 0.456, 0.406)  # Normalizeのmean値
    std:           tuple              = (0.229, 0.224, 0.225)  # Normalizeのstd値

    save_dir:      str                = 'results'              # 学習結果を保存するディレクトリ(exists_ok=True)
    train_dir:     str                = 'dataset/train'        # 学習用データがあるディレクトリ
    valid_dir:     str                = 'dataset/valid'        # バリデーション用データがあるディレクトリ
    test_dir:      str                = 'dataset/test'         # テスト用データがあるディレクトリ


@dataclass
class TrainDataBase:
    early_stop:    int                = 0                      # 精度が向上しなかった連続エポック数をカウント
    best_acc:      float              = 0.0000                 # 最も高かった精度の数値
    train_acc:     float              = 0.0000                 # 直近1エポックの精度
    valid_acc:     float              = 0.0000                 # 直近1バリデーションの精度
    train_loss:    float              = 0.0000                 # 直近1エポックの損失
    valid_loss:    float              = 0.0000                 # 直近1バリデーションの損失


class Config(ConfigBase):
    @staticmethod
    def get_device() -> torch.device:
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        return device

    def __repr__(self) -> str:
        distance: int = 12
        reprs: str = '=' * 15 + ' Config ' + '=' * 15 + '\n'

        for k, v in self.__dict__.items():
            length: int = distance - len(k)
            reprs += k + ' ' * length + ' : ' + str(v) + '\n'

        reprs += '=' * 38 + '\n'
        return reprs


class TrainData(TrainDataBase):
    def __repr__(self) -> str:
        ret: str = (f' - Train Accuracy: {self.train_acc:.4}'
                    f' - Train Loss: {self.train_loss:.4f}'
                    f' - Valid Accuracy: {self.valid_acc:.4}'
                    f' - Valid Loss: {self.valid_loss:.4f}'
                    f' - Best Accuracy: {self.best_acc:.4}'
                    f' - Early Stop: {self.early_stop}')
        return ret


class Labels:
    def __init__(self, my_num: int, classes: int):
        self.my_num: int = my_num
        self.classes: int = classes
        self.labels: dict = {i: 0 for i in range(classes)}

    def __call__(self, value: int):
        if value in self.labels:
            self.labels[value] += 1

    def __repr__(self) -> str:
        if 999 < self.classes:
            return ''

        w: int = 8
        s: str = ' '
        r: str = '\n'
        sep: str = '|'
        sep_h: str = '||'
        base: str = 'Label'

        def header() -> str:
            ret: str = s * (w+1) + sep_h
            for i in range(len(self.labels)):
                l: int = w - len(str(i)) - len(base)
                ret += s * l + base + s + str(i) + sep
            ret += r
            return ret

        def separator():
            ret: str = '=' * 10 * (len(self.labels) + 1) + '='
            ret += r
            return ret

        def body(my_num: int) -> str:
            l: int = w - len(str(my_num)) - len(base)
            ret: str = base + s + str(my_num) + s * l + sep_h
            for v in self.labels.values():
                l: int = (w+1) - len(str(v))
                ret += s * l + str(v) + sep
            return ret

        reprs = ''
        if not bool(self.my_num):
            reprs += header()
        reprs += separator()
        reprs += body(self.my_num)

        return reprs
