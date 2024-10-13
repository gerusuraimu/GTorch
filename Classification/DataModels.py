from typing import List, Tuple, Optional
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class ConfigBase:
    architecture:  Optional[str]          = None                   # 使用アーキテクチャ名
    results:       Optional[str]          = None                   # 訓練したモデルを保存しているディレクトリ
    model:         Optional[str]          = None                   # 保存される（読み込んだ）モデルのファイル名
    imgsz:         Tuple[int]             = (320, 320)             # 入力画像のサイズ ... (H, W)
    classes:       Optional[int]          = None                   # 分類クラス数

    epoch:         int                    = 100                    # 学習エポック数
    batch:         int                    = 64                     # 学習バッチサイズ
    worker:        int                    = 2                      # 学習時ワーカー数
    lr:            float                  = 0.0001                 # 学習率
    weights:       bool                   = True                   # PyTorchの事前学習済み重みを使用するかどうか
    early_stop:    int                    = 50                     # 学習中に精度が向上しなかった場合に早期学習終了を判断するエポック数

    device:        Optional[torch.device] = None                   # 学習に使用するデバイス(CUDA, MPS, CPU)
    pin_memory:    bool                   = True                   # ピンメモリーを使用するかどうか
    mean:          Tuple[float]           = (0.485, 0.456, 0.406)  # Normalizeのmean値
    std:           Tuple[float]           = (0.229, 0.224, 0.225)  # Normalizeのstd値

    save_dir:      str                    = 'results'              # 学習結果を保存するディレクトリ(exists_ok=False)
    train_dir:     str                    = 'dataset/train'        # 学習用データがあるディレクトリ
    valid_dir:     str                    = 'dataset/valid'        # バリデーション用データがあるディレクトリ
    test_dir:      str                    = 'dataset/test'         # テスト用データがあるディレクトリ


@dataclass
class TrainDataBase:
    early_stop:    int                    = 0                      # 精度が向上しなかった連続エポック数をカウント
    best_acc:      float                  = 0.0000                 # 最も高かった精度
    best_loss:     float                  = np.inf                 # best_accの時に最も低かった損失
    train_acc:     float                  = 0.0000                 # 直近1エポックの精度
    valid_acc:     float                  = 0.0000                 # 直近1バリデーションの精度
    train_loss:    float                  = 0.0000                 # 直近1エポックの損失
    valid_loss:    float                  = 0.0000                 # 直近1バリデーションの損失


class Config(ConfigBase):
    @staticmethod
    def get_device() -> torch.device:
        device: torch.device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        return device

    def __repr__(self) -> str:
        distance: int = 12
        separator: str = '=' * 38
        header: str = f"{'=' * 15} Config {'=' * 15}\n"

        body: str = '\n'.join(f"{key.ljust(distance)} : {value}" for key, value in self.__dict__.items())
        reprs: str = f"{header}{body}\n{separator}\n"
        return reprs


class TrainData(TrainDataBase):
    def __repr__(self) -> str:
        ret: str = (f' - Train Accuracy: {self.train_acc:.4f}'
                    f' - Train Loss: {self.train_loss:.4f}'
                    f' - Valid Accuracy: {self.valid_acc:.4f}'
                    f' - Valid Loss: {self.valid_loss:.4f}'
                    f' - Best Accuracy: {self.best_acc:.4f}'
                    f' - Best Loss: {self.best_loss:.4f}'
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
        if self.classes > 999:
            return ''

        w: int = 8
        sep: str = '|'
        sep_h: str = '||'
        base: str = 'Label'

        def header() -> str:
            part: List[str] = [' ' * (w + 1) + sep_h]
            for i in range(len(self.labels)):
                label: str = f" {base} {i} "
                part.append(label.rjust(w) + sep)
            return ''.join(part) + '\n'

        def separator() -> str:
            return '=' * (10 * (len(self.labels) + 1) + 1) + '\n'

        def body(my_num: int) -> str:
            part: List[str] = [f"{base} {my_num}".ljust(w + 1) + sep_h]
            for value in self.labels.values():
                part.append(str(value).rjust(w + 1) + sep)
            return ''.join(part)

        parts: List[str] = []
        if not self.my_num:
            parts.append(header())
        parts.append(separator())
        parts.append(body(self.my_num))

        return ''.join(parts)
