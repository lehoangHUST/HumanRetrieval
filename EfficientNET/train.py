import os
from tqdm import tqdm
import shutil
import time
import yaml
import argparse
import pprint
pp = pprint.PrettyPrinter()

import torch
import numpy as np

from modeling.model import Model
from utils.loss import Loss
from utils.dataset import create_dataloader, plot_images
from utils.utils import get_imgsz, select_device
from utils.metrics import accuracy, AverageMeter, fitness
import val

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# TODO: learning rate schedule


def save_ckpt(save_dir, state, is_best=False, filename="last.pt"):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_place = save_dir + '/' + filename
    torch.save(state, save_place)
    if is_best:
        shutil.copyfile(save_place, save_dir + '/' + "best.pt")


def run(args):
    device, dataset, augmentations, lsCE, lsBCE, batch_size, workers, \
    extractor, pretrained, resume, weight, epochs, save_dir = \
          args.device, args.dataset, args.augmentation, args.label_smoothing_CE, args.label_smoothing_BCE, \
          args.batch_size, args.workers, args.extractor, args.pretrained, args.resume, args.weight, args.epochs, args.save_dir

    s = "Training with "
    slsCE = "Cross Entropy Label Smoothing" if lsCE else ""
    slsBCE = "Binary Cross Entropy Label Smoothing" if lsBCE else ""
    s = f"{s} {slsBCE} {slsCE}"
    print(s)

    # initialize
    device = select_device(device)
    imgsz = get_imgsz(extractor) # set image size to match with efficient-net scaling
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # from yaml
    with open(dataset) as f:
        dataset = yaml.safe_load(f) # datadict
    if augmentations is not None:
        with open(augmentations) as f:
            augmentations = yaml.safe_load(f) # augmentation hyps
            pp.pprint(augmentations)


    # dataset
    train_loader, train_dataset = create_dataloader(dataset, imgsz, batch_size, workers, task='train',
                                                    augment=True if augmentations is not None else False, augment_config=augmentations)
    if not args.noval:
        val_loader, val_dataset = create_dataloader(dataset, imgsz, batch_size, workers, task='val',
                                                    augment=False, augment_config=None)

    # model
    if pretrained:
        print(f"Using pre-trained model {extractor}")
        model = Model(extractor,
                      True,
                      train_dataset.type_len,
                      train_dataset.color_len).to(device)
    else:
        print(f"Creating model {extractor}")
        model = Model(extractor,
                      False,
                      train_dataset.type_len,
                      train_dataset.color_len).to(device)

    # loss
    loss = Loss(train_dataset.type_len, label_smoothing0=lsCE, label_smoothing1=lsBCE)

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

        # construct metrics
        losses = AverageMeter()
        type_losses = AverageMeter()
        color_losses = AverageMeter()
        acc1 = AverageMeter()  # acc for clothes type
        correct_colors = torch.tensor([0] * train_dataset.color_len)  # number of correct color predictions

        model.train() # set the model to training mode

        start = time.time()

        for i, sample in enumerate(tqdm(train_loader, desc="Training batch", unit='batch')):
            # plot training batch for first 3 batch and only at the first epoch
            if i < 3 and not resume and epoch == 0:
                plot_images(samples=sample, save_folder=save_dir, fname=f"train_batch{i+1}.jpg")

            # prepare targets
            inputs = (sample["image"] / 255.).to(device) # torch.Tensor
            type_targets = sample["type_onehot"] # torch.Tensor
            color_targets = sample["color_onehot"] # torch.Tensor
            targets = torch.cat([type_targets, color_targets], dim=1).to(device) # torch.Tensor

            # compute outputs
            outputs = model(inputs)  # [type_pred: torch.Tensor, color_pred: torch.Tensor]
            total_loss, type_loss, color_loss = loss(outputs, targets) # torch.Tensor

            # accuracy and loss
            # type_acc: torch.Tensor on device
            # color_matching: torch.Tensor on cpu
            type_acc, color_matching = accuracy(outputs, targets, train_dataset)
            acc1.update(type_acc.item(), inputs.size(0))
            correct_colors += color_matching
            losses.update(total_loss.item(), inputs.size(0))
            type_losses.update(type_loss.item(), inputs.size(0))
            color_losses.update(color_loss.item(), inputs.size(0))

            # compute gradient and optimizer step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # compute color acc
        num_color_dict = train_dataset.get_color_statistic()
        total_color = np.array(list(num_color_dict.values()))
        color_acc = (correct_colors / total_color) * 100
        avg_color_acc = torch.sum(color_acc) / train_dataset.color_len

        end = time.time()
        epoch_time = end - start
        print(f'Epoch: {epoch + 1} \t Time: {epoch_time}s')

        # logging
        s = ""
        for i in range(train_dataset.color_len):
            s += f'{train_dataset.clothes_color[i]} acc: {color_acc[i]:.4f} \t'
        print(f'Total loss: {losses.avg:.4f} \t'
              f'Type loss: {type_losses.avg:.4f} \t'
              f'Color loss: {color_losses.avg:.4f} \t'
              f'Type acc: {acc1.avg:.4f} \t'
              f'Avg color acc: {avg_color_acc:.4f} \n'
              f'{s} \n')

        if not args.noval:
            losses, type_loss, color_loss, acc1, avg_color_acc = val.run(
                val_dataset, val_loader, device=device, loss=loss, model=model, epoch=epoch
            )

        # metrics for monitoring
        metrics = np.array((losses.avg, type_losses.avg, color_losses.avg, acc1.avg, avg_color_acc))

        # save checkpoint
        fi = fitness(metrics.flatten())
        if fi > best_fitness:
            best_fitness = fi
            print(f"Saving best model with fitness {fi:.4f}")
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckpt(save_dir, state, is_best=True)
        else:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckpt(save_dir, state)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dataset', default="config/dataset.yaml", type=str, help='path to dataset.yaml')
    parser.add_argument('--augmentation', default=None, type=str, help='path to augmentation.yaml')
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
    parser.add_argument('--save_dir', type=str, default='saved_model', help='path to save your training model weights')
    parser.add_argument('--noval', action='store_true', help="flag to set if don't want to evaluate on a validation set")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)

