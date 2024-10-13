import os
import csv
from _csv import writer
import pickle
from typing import Any

from tqdm import tqdm
import torch
from torch.optim import RAdam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose

from . import Utils
from .DataModels import Config, TrainData


class Trainer:
    def __init__(self, config: Config, model: Any):
        self.model:         Any              = model
        self.config:        Config           = config
        self.transform:     Compose          = Utils.get_transform(config)
        self.data:          TrainData        = TrainData()
        self.es_counter:    int              = 0
        self.best_accuracy: float            = 0.0
        self.test_accuracy: float            = 0.0
        self.criterion:     CrossEntropyLoss = CrossEntropyLoss()
        self.optimizer:     RAdam            = RAdam(self.model.parameters(), lr=self.config.lr)

        self.train_dataset: ImageFolder      = self.get_dataset(self.config.train_dir)
        self.valid_dataset: ImageFolder      = self.get_dataset(self.config.valid_dir)
        self.train_loader:  DataLoader       = self.get_dataloader(self.train_dataset)
        self.valid_loader:  DataLoader       = self.get_dataloader(self.valid_dataset)

        self.model.to(self.config.device)

        os.makedirs(self.config.save_dir, exist_ok=False)

    def train(self):
        train_accuracy: float
        valid_accuracy: float
        train_loss:     float
        valid_loss:     float
        log_name:       str   = os.path.join(self.config.save_dir, 'log.txt')
        best_name:      str   = os.path.join(self.config.save_dir, 'model.pt')
        pickle_name:    str   = os.path.join(self.config.save_dir, 'config.pickle')
        self.save_config(pickle_name)

        with open(log_name, 'w') as f:
            log: writer = csv.writer(f)
            log.writerow(['Epoch', 'TrainAccuracy', 'TrainLoss', 'ValidAccuracy', 'ValidLoss'])  # logのヘッダー書き込み

            for epoch in range(self.config.epoch):
                self.training(epoch)          # 訓練実行
                self.validation(epoch)        # バリデーション実行
                self.refresh_best(best_name)  # 訓練・バリデーションの結果からベストスコアを更新
                self.write_log(log, epoch)    # ログ出力

                if self.es_counter == self.config.early_stop:
                    print('Early Stop!!')
                    break

    def training(self, epoch: int):
        results: torch.Tensor

        correct: float = 0.0
        loss: float = 0.0
        desc: str = f'Train {epoch + 1}/{self.config.epoch}'

        for images, labels in tqdm(self.train_loader, desc=desc):
            images: torch.Tensor = images.to(self.config.device)
            labels: torch.Tensor = labels.to(self.config.device)
            self.optimizer.zero_grad()

            outputs: torch.Tensor = self.model(images)
            results: torch.Tensor
            _, results = torch.max(outputs.data, 1)
            iter_loss: torch.Tensor = self.criterion(outputs, labels)
            iter_loss.backward()
            self.optimizer.step()

            correct += (results == labels).sum().item()
            loss += iter_loss.item()

        self.data.train_acc = correct / len(self.train_dataset)
        self.data.train_loss = loss / len(self.train_loader)

    def validation(self, epoch: int):
        results: torch.Tensor

        correct: float = 0.0
        loss: float = 0.0

        self.model.eval()

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images: torch.Tensor = images.to(self.config.device)
                labels: torch.Tensor = labels.to(self.config.device)

                outputs: torch.Tensor = self.model(images)
                _, results = torch.max(outputs.data, 1)
                iter_loss: torch.Tensor = self.criterion(outputs, labels)

                correct += (results == labels).sum().item()
                loss += iter_loss.item()
                self.data.valid_acc = correct / len(self.valid_dataset)
                self.data.valid_loss = loss / len(self.valid_loader)
                print(f'\rValid {epoch+1}/{self.config.epoch}', self.data, f'/{self.config.early_stop}', sep='', end='')
        print()

    def save_config(self, name: str):
        with open(name, 'wb') as pkl:
            pickle.dump(self.config, pkl)

    def get_dataset(self, _dir: str) -> ImageFolder:
        return ImageFolder(root=_dir, transform=self.transform)

    def get_dataloader(self, dataset: ImageFolder, shuffle: bool = True) -> DataLoader:
        worker: int = self.config.worker if 0 < self.config.worker < os.cpu_count() else os.cpu_count()
        dataloader: DataLoader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch,
            shuffle=shuffle,
            num_workers=worker,
            pin_memory=self.config.pin_memory
        )
        return dataloader

    def refresh_best(self, best_name: str):
        if self.data.best_acc < self.data.valid_acc:
            # 今回のエポックにおけるバリデーション精度が訓練全体で最も良い精度より高ければ、ベストモデル、精度、損失の値を更新。
            pass
        elif self.data.best_acc == self.data.valid_acc and self.data.valid_loss < self.data.best_loss:
            # 今回のエポックにおけるバリデーション精度が訓練全体で最も良い精度と同じでも損失が少ないなら、ベストモデル、損失の値を更新。
            pass
        else:
            self.data.early_stop += 1
            return

        self.data.best_acc = self.data.valid_acc
        self.data.best_loss = self.data.valid_loss
        torch.save(self.model.state_dict(), best_name)
        self.data.early_stop = 0

    def write_log(self, log: writer, epoch: int):
        log.writerow([
            epoch + 1,
            f'{self.data.train_acc:.4f}',
            f'{self.data.train_loss:.4f}',
            f'{self.data.valid_acc:.4f}',
            f'{self.data.valid_loss:.4f}'
        ])

