import os
from tqdm import tqdm
import shutil
import time
import yaml
import argparse

import torch
import numpy as np

from modeling.model_v2 import Model_type
from modeling.model_v2 import Model_color
from utils.loss_v2 import Loss_typ
from utils.loss_v2 import Loss_color
from utils.dataset import create_dataloader, ClothesClassificationDataset
from utils.utils import get_imgsz, select_device
from utils.metrics_v2 import accuracy_type, accuracy_color, AverageMeter, fitness_type

def run_type(dataset: ClothesClassificationDataset,
        dataloader,
        device,
        loss,
        weight="",
        extractor="",
        model: torch.nn.Module=None,
        epoch=None):

    """
        Type loss: CrossEntropy
    """

    if epoch:
        print(f"Validating on epoch {epoch + 1}")
    if epoch is None:
        print('=' * 30)
        print(f"Validating")
    # construct metrics
    type_losses = AverageMeter()
    acc1 = AverageMeter()  # acc for clothes type

    training = model is not None # check if called by train.py
    if training:
        device = next(model.parameters()).device
    else:
        device = select_device(device)

        # init model
        model = Model_type(extractor,
                      False,
                      dataset.type_len).to(device)

        # load weights
        checkpoint = torch.load(weight, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        correct_type = np.array([0] * dataset.type_len, dtype=np.int_)
        for sample in tqdm(dataloader, desc="Validation batch", unit='batch'):
            # prepare targets
            inputs = (sample["image"] / 255.).to(device) # torch.Tensor
            targets = sample["type_onehot"].to(device) # torch.Tensor

            # compute outputs
            outputs = model(inputs)  # type_predict: torch.Tensor
            type_loss = loss(outputs, targets) # torch.Tensor

            # accuracy and loss
            # type_acc: torch.Tensor on device
            # color_matching: torch.Tensor on cpu
            type_matching = accuracy_type(outputs, targets, dataset)
            correct_type += type_matching.numpy()
            type_losses.update(type_loss.item(), inputs.size(0))

        # compute color acc
        num_type_dict = dataset.get_type_statistic()
        total_type = np.array(list(num_type_dict.values()))
        type_acc = correct_type / total_type
        avg_type_acc = np.sum(type_acc) / dataset.type_len

        # logging
        s = ""
        for i in range(dataset.type_len):
            s += f'{dataset.clothes_type[i]} acc: {type_acc[i]:.4f} \t'
        print(f'Type loss: {type_losses.avg:.4f} \t'
              f'Type acc: {avg_type_acc:.4f} \t'
              f'{s} \n')
    return type_losses, acc1

def run_color(dataset: ClothesClassificationDataset,
        dataloader,
        device,
        loss,
        weight="",
        extractor="",
        model: torch.nn.Module=None,
        epoch=None):

    """
        Color loss: Binary CrossEntropy
    """

    if epoch:
        print(f"Validating on epoch {epoch + 1}")
    if epoch is None:
        print('=' * 30)
        print(f"Validating")
    # construct metrics
    color_losses = AverageMeter()
    correct_colors = np.array([0] * dataset.color_len, dtype=np.int_)  # number of correct color predictions

    training = model is not None # check if called by train.py
    if training:
        device = next(model.parameters()).device
    else:
        device = select_device(device)

        # init model
        model = Model_color(extractor,
                      False,
                      dataset.color_len).to(device)

        # load weights
        checkpoint = torch.load(weight, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Validation batch", unit='batch'):
            # prepare targets
            inputs = (sample["image"] / 255.).to(device) # torch.Tensor
            targets = sample["color_onehot"].to(device) # torch.Tensor

            # compute outputs
            outputs = model(inputs)  # [type_pred: torch.Tensor, color_pred: torch.Tensor]
            color_loss = loss(outputs, targets) # torch.Tensor

            # accuracy and loss
            color_matching = accuracy_color(outputs, targets, dataset)
            correct_colors += color_matching.numpy()
            color_losses.update(color_loss.item(), inputs.size(0))

        # compute color acc
        num_color_dict = dataset.get_color_statistic()
        total_color = np.array(list(num_color_dict.values()))
        color_acc = correct_colors / total_color
        avg_color_acc = np.sum(color_acc) / dataset.color_len

        # logging
        s = ""
        for i in range(dataset.color_len):
            s += f'{dataset.clothes_color[i]} acc: {color_acc[i]:.4f} \t'
        print(f'Color loss: {color_losses.avg:.4f} \t'
              f'Avg color acc: {avg_color_acc:.4f} \n'
              f'{s} \n')
    return color_losses, avg_color_acc