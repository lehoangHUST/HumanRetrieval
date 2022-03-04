import argparse
from tqdm import tqdm
import time
import shutil


import torch

from modeling.model import Model
from utils.loss import Loss
from utils.datasets import *
from Classification_dict import dict as cls_dict
from utils.utils import get_imgsz, select_device
from utils.metrics import accuracy, AverageMeter

import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

#TODO: loss, acc history

def save_ckpt(save_dir, state, is_best=False, filename="last.pt"):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_place = save_dir + '/' + filename
    torch.save(state, save_place)
    if is_best:
        shutil.copyfile(save_place, save_dir + '/' + "best.pt")


def train(dataset, train_loader, model, criterions, optimizer, epoch, device):
    print(f'Training on epoch {epoch + 1}')
    losses = AverageMeter()
    type_losses = AverageMeter()
    color_losses = AverageMeter()
    acc1 = AverageMeter() # acc for clothes type
    acc2 = torch.tensor([0] * dataset.color_len) # acc for all colors

    # switch to training mode
    model.train()

    start = time.time()
    for sample in tqdm(train_loader, desc="Training batch", leave=False, unit='batch'):
        # prepare targets
        inputs = (sample["image"] / 255.).to(device)
        type_targets = sample["type_onehot"]
        color_targets = sample["color_onehot"]
        targets = torch.cat([type_targets, color_targets], dim=1).to(device)

        # compute outputs
        outputs = model(inputs) # [type_pred, color_pred]
        total_loss, type_loss, color_loss = criterions(outputs, targets)

        # accuracy and loss
        type_acc, color_matching = accuracy(outputs, targets, dataset)
        acc1.update(type_acc.item(), inputs.size(0))
        acc2 += color_matching
        losses.update(total_loss.item(), inputs.size(0))
        type_losses.update(type_loss.item(), inputs.size(0))
        color_losses.update(color_loss.item(), inputs.size(0))

        # compute gradient and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # compute color acc
    num_color_dict = dataset.get_statistic()
    total_color = torch.tensor(list(num_color_dict.values()))
    color_acc = acc2 / total_color

    end = time.time()
    epoch_time = end - start
    print(f'Epoch: {epoch + 1} \t Time: {epoch_time}s')

    # logging
    s = ""
    for i in range(dataset.color_len):
        s += f'{dataset.clothes_color[i]} acc: {color_acc[i]:.4f} \t'
    print(f'Total loss: {losses.avg:.4f} \t'
          f'Type loss: {type_losses.avg:.4f} \t'
          f'Color loss: {color_losses.avg:.4f} \t'
          f'Acc: {acc1.avg:.4f} \n'
          f'{s}')
    return model, losses.avg # return metrics to save model


def val(dataset, val_loader, model, criterions, epoch, device):
    print(f"Validating on epoch {epoch + 1}")
    losses = AverageMeter()
    type_losses = AverageMeter()
    color_losses = AverageMeter()
    acc1 = AverageMeter()
    acc2 = torch.tensor([0] * dataset.color_len)

    model.eval()
    with torch.no_grad():
        for sample in tqdm(val_loader, desc="Validation batch", leave=False, unit='batch'):
            inputs = (sample["image"] / 255.).to(device)
            type_targets = sample["type_onehot"]
            color_targets = sample["color_onehot"]
            targets = torch.cat([type_targets, color_targets], dim=1).to(device)

            # compute outputs
            outputs = model(inputs)  # [type_pred, color_pred]
            total_loss, type_loss, color_loss = criterions(outputs, targets)

            type_acc,  color_matching = accuracy(outputs, targets, dataset)
            acc1.update(type_acc.item(), inputs.size(0))
            acc2 += color_matching
            losses.update(total_loss.item(), inputs.size(0))
            type_losses.update(type_loss.item(), inputs.size(0))
            color_losses.update(color_loss.item(), inputs.size(0))
        # compute color acc
        num_color_dict = dataset.get_statistic()
        total_color = torch.tensor(list(num_color_dict.values()))
        color_acc = acc2 / total_color
    s = ""
    for i in range(dataset.color_len):
        s += f'{dataset.clothes_color[i]} acc: {color_acc[i]:.4f} \t'
    print(f'Total loss: {losses.avg: .4f} \t Type loss: {type_losses.avg: .4f} \t Color loss: {color_losses.avg: .4f} \t'
          f'Type accuracy: {acc1.avg: .4f} \n {s}')
    return losses.avg


def run(args):
    # Init
    device = select_device(args.device)
    imgsz = get_imgsz(args.extractor)

    # Load data
    train_dataloader, train_dataset = create_dataloader(args.train_dir,
                                            args.train_csv_file,
                                            cls_dict,
                                            imgsz,
                                            args.batch_size,
                                            args.workers)
    val_dataloader, val_dataset = create_dataloader(args.val_dir,
                                            args.val_csv_file,
                                            cls_dict,
                                            imgsz,
                                            args.batch_size,
                                            args.workers)

    # Build model
    if args.pretrained:
        print(f"Using pre-trained model {args.extractor}")
        model = Model(args.extractor,
                      True,
                      train_dataset.type_len,
                      train_dataset.color_len).to(device)
    else:
        print(f"Creating model {args.extractor}")
        model = Model(args.extractor,
                      False,
                      train_dataset.type_len,
                      train_dataset.color_len).to(device)

    # Build loss
    loss = Loss(train_dataset.type_len)

    # Build optimizer
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # Resume from ckpt
    if args.resume:
        weights = args.weights
        print(f"Loading checkpoint from {weights}")
        checkpoint = torch.load(weights, map_location=device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint, start at epoch: {start_epoch}")
    else:
        start_epoch = 0

    print("Start training")
    print('='*30)
    best_loss = 100
    for epoch in range(start_epoch, args.num_epochs):
        model, epoch_loss = train(train_dataset, train_dataloader, model, loss, optimizer, epoch, device)
        if args.val_dir or args.vd:
            val_loss = val(val_dataset, val_dataloader, model, loss, epoch, device)
        monitor = val_loss if val_loss else epoch_loss
        if monitor < best_loss:
            print(f'Saving model with loss {monitor}')
            best_loss = monitor
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckpt(args.save_dir, state, is_best=True)
        else:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_ckpt(args.save_dir, state)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=2, help='number of workers')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--extractor', type=str, default='efficientnet-b0', help='base feature extractor')
    parser.add_argument('--pretrained', action='store_true', help='load EfficientNet with ImageNet pretrained weights')
    parser.add_argument('--resume', action='store_true', help='load entire model with your trained weights')
    parser.add_argument('--weights', type=str, required=False, help='path to your trained weights')
    parser.add_argument('-td', '--train_dir', type=str, required=True, help='training directory stores all your image folder')
    parser.add_argument('-tcf' ,'--train_csv_file', type=str, required=True, help='name of your csv file contains info of your training dataset')
    parser.add_argument('-vd', '--val_dir', type=str, required=False, help="validation directory stores all  your image folder")
    parser.add_argument('-vcf', '--val_csv_file', type=str, required=False, help='name of your csv file contains info of your validation dataset')
    parser.add_argument('--save_dir', type=str, required=False, help='path to save your training model weights')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
