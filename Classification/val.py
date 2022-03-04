import os
from tqdm import tqdm
import argparse
import numpy as np

import torch

from modeling.model import Model
from utils.loss import Loss
from utils.dataset import create_dataloader, ClothesClassificationDataset
from utils.utils import get_imgsz, select_device
from utils.metrics import accuracy, AverageMeter


def run(dataset: ClothesClassificationDataset,
        dataloader,
        device,
        loss,
        weight="",
        extractor="",
        model: torch.nn.Module=None,
        epoch=None):
    """
    Function should returns metrics for fitness:
    - total_loss
    - type_loss
    - color_loss
    - type_acc
    - avg_color_acc
    """
    if epoch:
        print(f"Validating on epoch {epoch + 1}")
    if epoch is None:
        print('=' * 30)
        print(f"Validating")
    # construct metrics
    losses = AverageMeter()
    type_losses = AverageMeter()
    color_losses = AverageMeter()
    acc1 = AverageMeter()  # acc for clothes type
    correct_colors = torch.tensor([0] * dataset.color_len)  # number of correct color predictions

    training = model is not None # check if called by train.py
    if training:
        device = next(model.parameters()).device
    else:
        device = select_device(device)

        # init model
        model = Model(extractor,
                      False,
                      dataset.type_len,
                      dataset.color_len).to(device)

        # load weights
        checkpoint = torch.load(weight, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    with torch.no_grad():
        for sample in tqdm(dataloader, desc="Validation batch", unit='batch'):
            # prepare targets
            inputs = (sample["image"] / 255.).to(device) # torch.Tensor
            type_targets = sample["type_onehot"] # torch.Tensor
            color_targets = sample["color_onehot"] # torch.Tensor
            targets = torch.cat([type_targets, color_targets], dim=1).to(device) # torch.Tensor

            # compute outputs
            outputs = model(inputs)  # [type_pred: torch.Tensor, color_pred: torch.Tensor]
            total_loss, type_loss, color_loss = loss(outputs, targets) # torch.Tensor

            # accuracy and loss
            type_acc, color_matching = accuracy(outputs, targets, dataset)
            acc1.update(type_acc.item(), inputs.size(0))
            correct_colors += color_matching
            losses.update(total_loss.item(), inputs.size(0))
            type_losses.update(type_loss.item(), inputs.size(0))
            color_losses.update(color_loss.item(), inputs.size(0))

        # compute color acc
        num_color_dict = dataset.get_color_statistic()
        total_color = np.array(list(num_color_dict.values()))
        color_acc = (correct_colors / total_color) * 100
        avg_color_acc = torch.sum(color_acc) / dataset.color_len

        # logging
        s = ""
        for i in range(dataset.color_len):
            s += f'{dataset.clothes_color[i]} acc: {color_acc[i]:.4f} \t'
        print(f'Total loss: {losses.avg:.4f} \t'
              f'Type loss: {type_losses.avg:.4f} \t'
              f'Color loss: {color_losses.avg:.4f} \t'
              f'Type acc: {acc1.avg:.4f} \t'
              f'Avg color acc: {avg_color_acc:.4f} \n'
              f'{s} \n')
    return losses, type_losses, color_losses, acc1, avg_color_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dataset', default="config/dataset.yaml", type=str, help='path to dataset.yaml')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of workers')
    parser.add_argument('--extractor', type=str, default='efficientnet-b0', help='base feature extractor')
    parser.add_argument('--weight', type=str, required=False, help='path to your trained weights')
    parser.add_argument('--task', type=str, default='val', help='dataset to eval, val or train')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # initialize
    device = select_device(args.device)
    imgsz = get_imgsz(args.extractor) # set image size to match with efficient-net scaling

    # dataset
    dataloader, dataset = create_dataloader(args.dataset, imgsz, args.batch_size, args.workers,
                                            task=args.task, augment=False, augment_config=None)

    # loss
    loss = Loss(dataset.type_len)

    run(dataset, dataloader, device, loss, args.weight, args.extractor)

