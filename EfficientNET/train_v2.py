import os
from tqdm import tqdm
import shutil
import time
import yaml
import argparse

import torch
import numpy as np
import albumentations as A

from modeling.model_v2 import Model_type
from modeling.model_v2 import Model_color
from utils.loss_v2 import Loss_typ
from utils.loss_v2 import Loss_color
from utils.dataset import create_dataloader
from utils.utils import get_imgsz, select_device
from utils.metrics_v2 import accuracy_type, accuracy_color, AverageMeter, fitness_type
import val_v2


import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# TODO: learning rate schedule
def save_ckpt(save_dir, state, mission, extractor, is_best=False):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = extractor + mission + '.pt'
    save_place = save_dir + '/' + filename
    torch.save(state, save_place)
    if is_best:
        shutil.copyfile(save_place, save_dir + '/' + "best_" + filename)


def run(args):
    device, dataset, augmentations,  lsCE, lsBCE, batch_size, workers,  \
    extractor, pretrained, resume, weight, epochs, save_dir, mission = \
          args.device, args.dataset, args.augmentation, args.label_smoothing_CE, args.label_smoothing_BCE, args.batch_size, args.workers, \
          args.extractor, args.pretrained, args.resume, args.weight, args.epochs, args.save_dir, args.mission

    # Label smoothing or not label smoothing
    s = "Training with "
    slsCE = "Cross Entropy Label Smoothing" if lsCE else ""
    slsBCE = "Binary Cross Entropy Label Smoothing" if lsBCE else ""
    s = f"{s} {slsBCE} {slsCE}"
    print(s)

    # initialize
    device = select_device(device)
    imgsz = get_imgsz(extractor) # set image size to match with efficient-net scaling
    if not os.path.isdir(save_dir):
      os.mkdir(save_dir)

    # from yaml
    with open(dataset) as f:
        dataset = yaml.safe_load(f) # datadict
    with open(augmentations) as f:
        augmentations = yaml.safe_load(f) # augmentation hyps

    # dataset
    train_loader, train_dataset = create_dataloader(dataset, imgsz, batch_size, workers, task='train',
                                                    augment=True, augment_config=augmentations)
    if not args.noval:
        val_loader, val_dataset = create_dataloader(dataset, imgsz, batch_size, workers, task='val',
                                                    augment=False, augment_config=None)

    # Type or color clothes
    if mission == 'type_clothes':
      train_type(device, extractor, pretrained, resume, weight, epochs, save_dir, train_loader, train_dataset, val_loader, val_dataset, lsCE, mission)
    elif mission == 'color_clothes':
      train_color(device, extractor, pretrained, resume, weight, epochs, save_dir, train_loader, train_dataset, val_loader, val_dataset, lsBCE, mission)
    else:
      print(f"Not found mission not found.")

# Train type clothes
def train_type(device, extractor, pretrained, resume, weight, epochs, save_dir, train_loader, train_dataset, val_loader, val_dataset, lsCE, mission):
    # model type clothes
    if pretrained:
        print(f"Using pre-trained model {extractor}")
        model = Model_type(extractor,
                      True,
                      train_dataset.type_len).to(device)
    else:
        print(f"Creating model {extractor}")
        model = Model_type(extractor,
                      False,
                      train_dataset.type_len).to(device)

    # loss
    loss = Loss_typ(lsCE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # resume
    if resume:
        print(f"Loading checkpoint from {weight}")
        checkpoint = torch.load(weight, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint, start at epoch: {start_epoch + 1}")
    else:
        start_epoch = 0

    print("Start training")
    print('='*30 + '\n')

    best_fitness = 0.0

    for epoch in range(start_epoch, epochs):
        print('='*30)
        print(f"Start training on epoch {epoch + 1}")

        type_losses = AverageMeter()

        model.train() # set the model to training mode
        correct_type = np.array([0] * train_dataset.type_len, dtype=np.int)
        start = time.time()

        for sample in tqdm(train_loader, desc="Training batch", unit='batch'):

            # prepare targets
            inputs = (sample["image"] / 255.).to(device) # torch.Tensor
            targets = sample["type_onehot"].to(device) # torch.Tensor

            # compute outputs
            outputs = model(inputs)  # type_predict: torch.Tensor
            type_loss = loss(outputs, targets) # torch.Tensor

            # accuracy and loss
            # type_acc: torch.Tensor on device
            # types_acc: torch.Tensor(batch_size), type_acc: torch.Tensor(batch_size, dataset.type_len)
            type_matching = accuracy_type(outputs, targets, train_dataset) 
            
            correct_type += type_matching.numpy()
            type_losses.update(type_loss.item(), inputs.size(0))

            # compute gradient and optimizer step
            optimizer.zero_grad()
            type_loss.backward()
            optimizer.step()

        # compute type acc
        num_type_dict = train_dataset.get_type_statistic()
        total_type = np.array(list(num_type_dict.values()))
        type_acc = (correct_type / total_type) * 100
        avg_type_acc = np.sum(type_acc) / train_dataset.type_len

        end = time.time()
        epoch_time = end - start
        print(f'Epoch: {epoch + 1} \t Time: {epoch_time}s')

        # logging
        s = ""
        for i in range(train_dataset.type_len):
            s += f'{train_dataset.clothes_type[i]} acc: {type_acc[i]:.4f} \t'
        print(f'Type loss: {type_losses.avg:.4f} \t'
              f'Type acc: {avg_type_acc:.4f} \t'
              f'{s} \n')

        if not args.noval:
            print(f"Validating on epoch {epoch + 1}")
            type_loss, acc1 = val_v2.run_type(
                val_dataset, val_loader, device=device, loss=loss, model=model, epoch=epoch
            )

        # metrics for monitoring
        metrics = np.array((type_losses.avg, avg_type_acc))

        # save checkpoint
        fi = fitness_type(metrics.flatten())
        if fi > best_fitness:
            best_fitness = fi
            print(f"Saving best model with fitness {fi:.4f}")
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckpt(save_dir, state, mission, extractor, is_best=True)
        else:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckpt(save_dir, state, mission, extractor)

# Train type color clothes
# Train type clothes
def train_color(device, extractor, pretrained, resume, weight, epochs, save_dir, train_loader, train_dataset, val_loader, val_dataset, lsBCE, mission):
    # model type clothes
    if pretrained:
        print(f"Using pre-trained model {extractor}")
        model = Model_color(extractor,
                      True,
                      train_dataset.color_len).to(device)
    else:
        print(f"Creating model {extractor}")
        model = Model_color(extractor,
                      False,
                      train_dataset.color_len).to(device)

    # loss
    loss = Loss_color(lsBCE)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # resume
    if resume:
        print(f"Loading checkpoint from {weight}")
        checkpoint = torch.load(weight, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint, start at epoch: {start_epoch + 1}")
    else:
        start_epoch = 0

    print("Start training")
    print('='*30 + '\n')

    best_fitness = 0.0

    for epoch in range(start_epoch, epochs):
        print('='*30)
        print(f"Start training on epoch {epoch + 1}")

        color_losses = AverageMeter()
        correct_colors = np.array([0] * train_dataset.color_len, dtype=np.int)  # number of correct color predictions

        model.train() # set the model to training mode

        start = time.time()

        for sample in tqdm(train_loader, desc="Training batch", unit='batch'):
            # prepare targets
            inputs = (sample["image"] / 255.).to(device) # torch.Tensor
            targets = sample["color_onehot"].to(device) # torch.Tensor

            # compute outputs
            outputs = model(inputs)  # type_predict: torch.Tensor
            color_loss = loss(outputs, targets) # torch.Tensor

            # accuracy and loss
            # color_matching: torch.Tensor on cpu
            color_matching = accuracy_color(outputs, targets, train_dataset)
            correct_colors += color_matching.numpy()
            color_losses.update(color_loss.item(), inputs.size(0))
          
            # compute gradient and optimizer step
            optimizer.zero_grad()
            color_loss.backward()
            optimizer.step()

        # compute color acc
        num_color_dict = train_dataset.get_color_statistic()
        total_color = np.array(list(num_color_dict.values()))
        color_acc = (correct_colors / total_color) * 100
        avg_color_acc = np.sum(color_acc) / train_dataset.color_len

        end = time.time()
        epoch_time = end - start
        print(f'Epoch: {epoch + 1} \t Time: {epoch_time}s')

        # logging
        s = ""
        for i in range(train_dataset.color_len):
            s += f'{train_dataset.clothes_color[i]} acc: {color_acc[i]:.4f} \t'
        print(f'Color loss: {color_losses.avg:.4f} \t'
              f'Avg color acc: {avg_color_acc:.4f} \n'
              f'{s} \n')

        if not args.noval:
            print(f"Validating on epoch {epoch + 1}")
            color_loss, avg_color_acc = val_v2.run_color(
                val_dataset, val_loader, device=device, loss=loss, model=model, epoch=epoch
            )

        # metrics for monitoring
        metrics = np.array((color_loss.avg, avg_color_acc))

        # save checkpoint
        fi = fitness_type(metrics.flatten())
        if fi > best_fitness:
            best_fitness = fi
            print(f"Saving best model with fitness {fi:.4f}")
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckpt(save_dir, state, mission, extractor, is_best=True)
        else:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckpt(save_dir, state, mission, extractor)

# Parse argument
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dataset', default="/content/gdrive/MyDrive/HumanRetrieval_v2/EfficientNET/config/dataset.yaml", type=str, help='path to dataset.yaml')
    parser.add_argument('--augmentation', default="/content/gdrive/MyDrive/HumanRetrieval_v2/EfficientNET/config/augmentation.yaml", type=str, help='path to augmentation.yaml')
    parser.add_argument('--label_smoothing_CE', '-lsCE', action='store_true', help='use label smoothing on CrossEntropy')
    parser.add_argument('--label_smoothing_BCE', '-lsBCE', action='store_true', help='use label smoothing on BCE')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of workers')
    parser.add_argument('--extractor', type=str, default='efficientnet-b0', help='base feature extractor')
    parser.add_argument('--pretrained', action='store_true', help='load EfficientNet with ImageNet pretrained weights')
    parser.add_argument('--resume', action='store_true', help='load entire model with your trained weights')
    parser.add_argument('--weight', type=str, required=False, help='path to your trained weights')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--fitness_weight', '-fn', )
    parser.add_argument('--save_dir', type=str, default='/content/gdrive/MyDrive/model/', required=False, help='path to save your training model weights')
    parser.add_argument('--noval', action='store_true', help="flag to set if don't want to evaluate on a validation set")
    parser.add_argument('--mission', default='type_clothes', type=str, help='This missions purpose is to train type or color clothes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)

