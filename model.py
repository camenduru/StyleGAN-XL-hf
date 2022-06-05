from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

current_dir = pathlib.Path(__file__).parent
submodule_dir = current_dir / 'stylegan_xl'
sys.path.insert(0, submodule_dir.as_posix())

HF_TOKEN = os.environ['HF_TOKEN']


class Model:

    MODEL_NAMES = [
        'imagenet16',
        'imagenet32',
        'imagenet64',
        'imagenet128',
        'cifar10',
        'ffhq256',
        'pokemon256',
    ]

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self._download_all_models()
        self.model_name = self.MODEL_NAMES[3]
        self.model = self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> nn.Module:
        path = hf_hub_download('hysts/StyleGAN-XL',
                               f'models/{model_name}.pkl',
                               use_auth_token=HF_TOKEN)
        with open(path, 'rb') as f:
            model = pickle.load(f)['G_ema']
        model.eval()
        model.to(self.device)
        return model

    def set_model(self, model_name: str) -> None:
        if model_name == self.model_name:
            return
        self.model_name = model_name
        self.model = self._load_model(model_name)

    def _download_all_models(self):
        for name in self.MODEL_NAMES:
            self._load_model(name)

    @staticmethod
    def make_transform(translate: tuple[float, float],
                       angle: float) -> np.ndarray:
        mat = np.eye(3)
        sin = np.sin(angle / 360 * np.pi * 2)
        cos = np.cos(angle / 360 * np.pi * 2)
        mat[0][0] = cos
        mat[0][1] = sin
        mat[0][2] = translate[0]
        mat[1][0] = -sin
        mat[1][1] = cos
        mat[1][2] = translate[1]
        return mat

    def generate_z(self, seed: int) -> torch.Tensor:
        seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
        z = np.random.RandomState(seed).randn(1, self.model.z_dim)
        return torch.from_numpy(z).float().to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(
            torch.uint8)
        return tensor.cpu().numpy()

    def make_label_tensor(self, class_index: int) -> torch.Tensor:
        class_index = round(class_index)
        class_index = min(max(0, class_index), self.model.c_dim - 1)
        class_index = torch.tensor(class_index, dtype=torch.long)

        label = torch.zeros([1, self.model.c_dim], device=self.device)
        if class_index >= 0:
            label[:, class_index] = 1
        return label

    def set_transform(self, tx: float, ty: float, angle: float) -> None:
        mat = self.make_transform((tx, ty), angle)
        mat = np.linalg.inv(mat)
        self.model.synthesis.input.transform.copy_(torch.from_numpy(mat))

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, label: torch.Tensor,
                 truncation_psi: float) -> torch.Tensor:
        return self.model(z, label, truncation_psi=truncation_psi)

    def generate_image(self, seed: int, truncation_psi: float,
                       class_index: int, tx: float, ty: float,
                       angle: float) -> np.ndarray:
        self.set_transform(tx, ty, angle)

        z = self.generate_z(seed)
        label = self.make_label_tensor(class_index)

        out = self.generate(z, label, truncation_psi)
        out = self.postprocess(out)
        return out[0]

    def set_model_and_generate_image(self, model_name: str, seed: int,
                                     truncation_psi: float, class_index: int,
                                     tx: float, ty: float,
                                     angle: float) -> np.ndarray:
        self.set_model(model_name)
        return self.generate_image(seed, truncation_psi, class_index, tx, ty,
                                   angle)
