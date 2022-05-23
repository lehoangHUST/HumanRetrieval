from modeling.model_v2 import Model_type, Model_color
from Classification_dict import dict as clsdict
from utils.utils import select_device, get_imgsz, convert_output
from utils.dataset import LoadImage

import torch
import yaml
from torchvision import transforms
from utils import utils
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse


@torch.no_grad()
def run(args):
    device = select_device(args.device)
    # datadict for model efficientNet
    with open(args.cls_data) as f:
      cls_dataset = yaml.safe_load(f) 
    imgsz = get_imgsz(args.base_model)
    net_type = Model_type(args.base_model,
                  use_pretrained=False,
                  num_class=len(cls_dataset['class']['Type']))
    net_type.load_state_dict(torch.load(args.type_clothes_weight)['state_dict'])
    net_type.to(device)
    net_type.eval()

    # Color clothes 
    net_color = Model_color(args.base_model,
                  use_pretrained=False,
                  num_class=len(cls_dataset['class']['Color']))
    net_color.load_state_dict(torch.load(args.color_clothes_weight)['state_dict'])
    net_color.to(device)
    net_color.eval()

    dataset = LoadImage(args.image_path)
    for path, im in dataset:
        image = net_type.preprocess(im.copy())
        #image = torch.from_numpy(np.expand_dims(orig_image[:, :, ::-1].transpose(2, 0, 1), axis=0).copy()).to(device) #convert to (batch, c, h, w)
        #image = transforms.Resize(imgsz)(image) / 255.

        clothes_output = net_type(image)
        color_output = net_color(image)
        # type_pred: string
        # color_pred: list(string)
        type_pred, color_pred = utils.convert_output(cls_dataset['class'], [clothes_output, color_output])
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--base_model', default='efficientnet-b1', type=str, help='feature extractor')
    parser.add_argument('--type_clothes_weight', type=str, default="/content/gdrive/MyDrive/HumanRetrieval_v2/train/best_efficientnet-b1type_clothes.pt")
    parser.add_argument('--cls_data', type=str, default="/content/gdrive/MyDrive/HumanRetrieval_v2/EfficientNET/config/dataset.yaml")
    parser.add_argument('--color_clothes_weight', type=str, default="/content/gdrive/MyDrive/HumanRetrieval_v2/train/best_efficientnet-b1color_clothes.pt")
    parser.add_argument('--image_path', required=True, type=str, help='path to image')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)

