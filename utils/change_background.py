# outside lib import
import os, sys
import argparse
import numpy as np
import cv2
import shutil
import yaml

import torch
import torch.backends.cudnn as cudnn

# own lib import
import modules
from Detection.yolov5.utils.datasets import LoadImages, LoadStreams, IMG_FORMATS
from Detection.yolov5.utils.torch_utils import time_sync, select_device
from Detection.yolov5.utils.general import set_logging, non_max_suppression, xyxy2xywh, scale_coords
from Detection.yolov5.utils.plots import Annotator

from Detection.eval_clothes import run_eval_clothes


@torch.no_grad()
def run(args):

    # Retrieval clothes 
    if ',' in args.clothes:
        clothes = args.clothes.split(',')
    else:
        clothes = [args.clothes]

    # Create dir to save image change background
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir) 

    # Load nets
    net_YOLACT, yolact_name = modules.config_Yolact(args.yolact_weight)
    
    for img_path in os.listdir(args.images):
        image_name = img_path.split('.')[0]
        img = cv2.imread(os.path.join(args.images, img_path))
        
        yolact_preds_bbox, yolact_preds_mask = run_eval_clothes(net_YOLACT,
                                        search_clothes=clothes,
                                        img_numpy=img)
        print(image_name)
        print("================================")
        print(yolact_preds_bbox)
        print("================================")
        if not isinstance(yolact_preds_bbox, int):
            # changebackground by color
            # =====================================================
            bbox = yolact_preds_bbox.type(torch.int32).cpu().numpy()
            yolact_preds_mask = yolact_preds_mask.type(torch.uint8).cpu().numpy()
            mask_clothes = []
            for i, mask in enumerate(yolact_preds_mask):
                for j in range(bbox[i][1], bbox[i][3]):
                    for k in range(bbox[i][0], bbox[i][2]):
                        if mask[j, k] == 0:
                          img[j, k, :] = 255
                mask_clothes.append(img[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2], :])
                cv2.imwrite(args.save_dir + '/' + image_name + '.jpg', img[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2], :])
        
    print(len(os.listdir(args.save_dir)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--yolact_weight', type=str, default="/content/gdrive/MyDrive/model/yolact_plus_resnet50_6_144000.pth")
    parser.add_argument('--clothes', type=str, default='short_sleeved_shirt')
    parser.add_argument('--save_dir', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)