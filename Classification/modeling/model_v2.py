from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils

import numpy as np
import os
import yaml

import torch
import torch.nn as nn
from torchvision import transforms


# Model for type clothes
class Model_type(nn.Module):
    def __init__(self, base_model: str,
                 use_pretrained:bool,
                 num_class,
                 dropout_rate=0.2):
        super(Model_type, self).__init__()
        if use_pretrained:
            self.base_model = EfficientNet.from_pretrained(base_model, include_top=False)
        else:
            self.base_model = EfficientNet.from_name(base_model, include_top=False)
        self.out_channel = utils.round_filters(1280, self.base_model._global_params)
        self._type_cls = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.out_channel, num_class)
        )
        self.imgsz = self.base_model.get_image_size(base_model)

    def forward(self, inputs):
        """ Perform forward pass on the input """
        x = self.base_model.extract_features(inputs)
        x = self.base_model._avg_pooling(x)
        x = x.flatten(start_dim=1)

        typ = self._type_cls(x)

        return typ

    def preprocess(self, imgs):
        """ Pre-process any given image or image batch to have desired input shape """
        #imgsz = get_imgsz(self.base_model)
        device = next(self.base_model.parameters()).device
        if isinstance(imgs, np.ndarray):
            imgs = imgs[:, :, ::-1] # BGR to RGB
            if imgs.ndim != 4:
                imgs = np.expand_dims(imgs, axis=0) # b, h, w, c
            imgs = imgs.transpose((0, 3, 1, 2)) # to b, c, h, w
            imgs = imgs.copy()
            imgs = torch.from_numpy(imgs) # torch.Tensor
        # sanity check
        assert isinstance(imgs, torch.Tensor), "input must be a numpy array or torch Tensor"
        assert imgs.dim() == 4, "Tensor must have shape (b, c, h, w)"

        imgs = transforms.Resize((self.imgsz, self.imgsz))(imgs)
        imgs = imgs / 255.
        imgs = imgs.to(device)
        return imgs

    def from_config(self, config):
        if os.path.isfile(config):
            with open(config) as f:
                cfg = yaml.safe_load(f)
        if isinstance(config, dict):
            cfg = config
        else:
            raise
        self.base_model = cfg["extractor"]
        self.num_cls1 = len(cfg["num_cls1"])
        self.num_csl2 = len(cfg["num_cls2"])

# Pred for type color in clothes
class Model_color(nn.Module):
    def __init__(self, base_model: str,
                 use_pretrained:bool,
                 num_class,
                 dropout_rate=0.2):
        super(Model_color, self).__init__()
        if use_pretrained:
            self.base_model = EfficientNet.from_pretrained(base_model, include_top=False)
        else:
            self.base_model = EfficientNet.from_name(base_model, include_top=False)
        self.out_channel = utils.round_filters(1280, self.base_model._global_params)
        self._color_cls = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.out_channel, num_class)
        )
        self.imgsz = self.base_model.get_image_size(base_model)

    def forward(self, inputs):
        """ Perform forward pass on the input """
        x = self.base_model.extract_features(inputs)
        x = self.base_model._avg_pooling(x)
        x = x.flatten(start_dim=1)

        color = torch.sigmoid(self._color_cls(x))

        return color

    def from_config(self, config):
        if os.path.isfile(config):
            with open(config) as f:
                cfg = yaml.safe_load(f)
        if isinstance(config, dict):
            cfg = config
        else:
            raise
        self.base_model = cfg["extractor"]
        self.num_cls1 = len(cfg["num_cls1"])
        self.num_csl2 = len(cfg["num_cls2"])