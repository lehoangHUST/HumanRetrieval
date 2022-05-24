# outside lib import
import os
import sys
import argparse
import numpy as np
import cv2
import shutil
import yaml

import torch
import torch.backends.cudnn as cudnn

# own lib import
import modules
from EfficientNET.modeling.model_v2 import Model_type
from EfficientNET.modeling.model_v2 import Model_color
from EfficientNET.modeling.model import Model
from EfficientNET.Classification_dict import dict as cls_dict
from EfficientNET.utils import utils
from Detection.yolov5.utils.datasets import LoadImages, LoadStreams, IMG_FORMATS
from Detection.yolov5.utils.torch_utils import time_sync, select_device
from Detection.yolov5.utils.general import set_logging, non_max_suppression, xyxy2xywh, scale_coords
from Detection.yolov5.utils.plots import Annotator

from Detection.eval_clothes import run_eval_clothes


classes = [
    "short_sleeved_shirt",
    "long_sleeved_shirt",
    "short_sleeved_outwear",
    "long_sleeved_outwear",
    "vest",
    "sling",
    "short",
    "trousers",
    "skirt",
    "short_sleeved_dress",
    "long_sleeved_dress",
    "vest_dress",
    "sling_dress"
]

top = [
    "short_sleeved_shirt",
    "long_sleeved_shirt",
    "short_sleeved_outwear",
    "long_sleeved_outwear",
]

bottom = [
    "short",
    "trousers",
    "skirt",
]


# Matching human and clothes
def matching(yolo_preds, yolact_preds_bbox, clothes):
    # A list of objects satisfying 2 properties
    list_det_human = []  # list of torch.Tensor containing bbox of human
    list_det_cls = []  # list of torch.Tensor containing bbox have mask of clothes

    if type(yolact_preds_bbox) == torch.Tensor and type(yolo_preds) == torch.Tensor:

        yolact_preds = yolact_preds_bbox.cpu().numpy()
        yolo_preds = yolo_preds.cpu().data.numpy()

        # Calculate inters set A and B
        def inters(bbox_a, bbox_b):
            # determine the coordinates of the intersection rectangle
            x_left = max(bbox_a[0], bbox_b[0])
            y_left = max(bbox_a[1], bbox_b[1])
            x_right = min(bbox_a[2], bbox_b[2])
            y_right = min(bbox_a[3], bbox_b[3])
            if x_right > x_left and y_right > y_left:
                return (x_right - x_left) * (y_right - y_left)
            else:
                return 0

        # Count = length of clothes: Draw bbox.
        # Count = not length of clothes: Not Draw bbox.
        for i in range(yolo_preds.shape[0]):
            count = 0
            bbox_clothes = {}
            for j in range(yolact_preds_bbox.shape[0]):
                # Calculate area.
                area_j = (yolact_preds[j][2] - yolact_preds[j][0]) * (
                        yolact_preds[j][3] - yolact_preds[j][1])
                area = inters(yolo_preds[i, :4], yolact_preds[j, :4])
                # Conditional
                if area / area_j > 0.7:
                    count += 1
                    if classes[int(yolact_preds_bbox[j, -1])] in top:
                        bbox_clothes['top'] = np.array(yolact_preds[j, :], dtype=np.int32).tolist()
                    elif classes[int(yolact_preds_bbox[j, -1])] in bottom:
                        bbox_clothes['bottom'] = np.array(yolact_preds[j, :], dtype=np.int32).tolist()

            if count == len(clothes):
                list_det_human.append(np.array(yolo_preds[i, :], dtype=np.float_).tolist())
                list_det_cls.append(bbox_clothes)
        return list_det_human, list_det_cls


@torch.no_grad()
def run(args):
    # Initialize
    set_logging()
    device = select_device(args.device)

    # datadict for model efficientNet
    with open(args.cls_data) as f:
        cls_dataset = yaml.safe_load(f)
        # TODO: map with config data instead of string processing
    humans = args.humans
    if len(humans) == 1:
        humans = [humans.index("".join(humans))]
    else:
        humans = [0, 1]  # With 0 is male and 1 is female.'''

    if ',' in args.clothes:
        clothes = args.clothes.split(',')
    else:
        clothes = [args.clothes]

    # Top: Torso of human
    if ',' in args.top:
        typ_top, color_top = args.top.split(',')
    else:
        raise TypeError

    # Bottom: Leg of human
    if ',' in args.bottom:
        typ_bottom, color_bottom = args.bottom.split(',')
    else:
        raise TypeError

    # disable for now
    '''if not all(elem in class_clothes for elem in search_clothes):
        raise ValueError(f"Have any category not exist in parameter classes")'''

    # Load nets YOLACT
    net_YOLACT, yolact_name = modules.config_Yolact(args.yolact_weight)

    # Load nets YOLOv5
    net_YOLO, strides, yolo_name, imgsz = modules.config_Yolov5(args.yolo_weight, device)

    # Load deepsort
    deepsort = modules.config_deepsort(args.cfg_deepsort, device)

    # Load net Type clothes
    net_type = Model_type(args.extractor,
                          use_pretrained=False,
                          num_class=len(cls_dataset['class']['Type']))
    net_type.load_state_dict(torch.load(args.type_clothes_weight)['state_dict'])
    net_type.to(device)
    net_type.eval()

    # Load net Color clothes
    net_color = Model_color(args.extractor,
                            use_pretrained=False,
                            num_class=len(cls_dataset['class']['Color']))
    net_color.load_state_dict(torch.load(args.color_clothes_weight)['state_dict'])
    net_color.to(device)
    net_color.eval()

    # Load data
    # Re-use yolov5 data loading pipeline for simplicity
    webcam = args.source.isnumeric()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(args.source, img_size=imgsz,
                              stride=strides)  # (sources, letterbox_img: np, orig_img: cv2, None)
    else:
        dataset = LoadImages(args.source, img_size=imgsz,
                             stride=strides)  # (path, letterbox_img: np, orig_img: cv2, cap)

    # saving prediction video
    if args.savevid:
        width = next(iter(dataset))[3].get(cv2.CAP_PROP_FRAME_WIDTH)
        height = next(iter(dataset))[3].get(cv2.CAP_PROP_FRAME_HEIGHT)
        res = (int(width), int(height))
        # this format fail to play in Chrome/Win10/Colab
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # codec
        # fourcc = cv2.VideoWriter_fourcc(*'H264') #codec
        output = cv2.VideoWriter(args.savename, fourcc, 30, res)

    # Run Inference
    for index, (path, im, im0s, vid_cap, _) in enumerate(dataset):
        det_sys = []
        human_label = ""
        is_img = True if any(ext in path for ext in IMG_FORMATS) else False
        annotator = Annotator(np.ascontiguousarray(im0s),
                              line_width=2,
                              font_size=1)

        # yolo inference
        # -----------------------------------------
        t1 = time_sync()
        im_yolo = torch.from_numpy(im).to(device)  # yolo input
        im_yolo = im_yolo.float()
        im_yolo /= 255
        if len(im_yolo.shape) == 3:
            im_yolo = im_yolo[None]  # expand for batch dim
        t2 = time_sync()
        # time logging for data loading
        dt0 = t2 - t1
        # Inference on yolov5
        yolo_preds = net_YOLO(im_yolo)  # (batch, (bbox, conf, class)) type torch.Tensor
        t3 = time_sync()
        # print(f"YOLO inference time: {t3 - t2:.4f}")
        # time logging for yolo predicting
        # nms for yolo
        # yolo_preds: torch.Tensor
        yolo_preds = non_max_suppression(yolo_preds, 0.6, 0.5, humans, None, max_det=100)[0]
        t4 = time_sync()
        # nms time for yolo
        # print(f"YOLO nms time: {t4 - t3:.4f}")

        # scale yolo preds to im0 for drawing
        if len(yolo_preds):
            yolo_preds[:, :4] = scale_coords(im_yolo.shape[2:], yolo_preds[:, :4], im0s.shape).round()

        # -----------------------------------------

        # yolact inference
        # -----------------------------------------
        # TODO: re-write the run_eval_clothes function, drop FastBaseTransform, drop prep_display
        im_yolact = im0s.copy()  # copy to another image so we can draw on im0s later
        # type torch.Tensor, shape (batch, (bbox, conf, cls))
        # type int if no detection
        t5 = time_sync()
        yolact_preds_bbox, yolact_preds_mask = run_eval_clothes(net_YOLACT,
                                                                search_clothes=clothes,
                                                                img_numpy=im_yolact)

        t6 = time_sync()
        # inference time for YOLACT
        # print(f"Inference time for YOLACT nms time: {t6 - t5:.4f}")

        if not isinstance(yolact_preds_bbox, int):
            # Matching yolact & yolov5
            # =====================================================
            t7 = time_sync()
            det_human, det_clothes_human = matching(yolo_preds, yolact_preds_bbox, clothes)
            t8 = time_sync()
            # inference time for matching yolact & yolov5
            # print(f"Inference time for YOLACT nms time: {t8 - t7:.4f}")
            # =====================================================

            # change background by color
            # =====================================================
            mask_det = []
            for det_cls in det_clothes_human:
                mask_clothes = {}
                for k, (body, bbox) in enumerate(det_cls.items()):
                    _img = im0s[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    img = np.ones(_img.shape, dtype=np.uint8) * 255
                    position = np.where(
                        yolact_preds_mask[k, bbox[1]:bbox[3], bbox[0]:bbox[2]].type(torch.uint8).cpu().numpy() == 1)
                    list_coordinate = list(zip(position[0], position[1]))
                    for coordinate in list_coordinate:
                        img[coordinate[0], coordinate[1], :] = _img[coordinate[0], coordinate[1], :]

                    if body == 'top':
                        mask_clothes['top'] = img
                    elif body == 'bottom':
                        mask_clothes['bottom'] = img

                mask_det.append(mask_clothes)

            # ======================================================
            # classification
            # ======================================================
            # 1. Read every single clothes ROI from yolact output one by one
            # 2. Perform preprocess to ROI
            # 3. Perform forward pass on image
            # 4. Convert output from classification model to correct read-able format
            # 5. Draw bbox with type and color label
            # =======================================================
            t9 = time_sync()
            clothes_labels = []
            for i, mask_clothes in enumerate(mask_det):
                dict_pred = {}
                for body, mask in mask_clothes.items():
                    inp = net_type.preprocess(mask)
                    color_output = net_color(inp)
                    if body == 'bottom':
                        colors = cls_dataset['class']['Color']
                        color_pred = []
                        color_output = (color_output > 0.5).type(torch.int)
                        color_output = color_output.cpu().detach().numpy()
                        for j in range(color_output.shape[1]):
                            if color_output[0, j] == 1:
                                color = colors[j]
                                color_pred.append(color)
                        idx = det_clothes_human[i]['bottom'][-1]
                        dict_pred['bottom'] = [classes[idx], color_pred, det_clothes_human[i]['bottom'][:4]]
                    if body == 'top':
                        clothes_output = net_type(inp)
                        # type_pred: string
                        # color_pred: list(string)
                        type_pred, color_pred = utils.convert_output(cls_dataset['class'],
                                                                     [clothes_output, color_output])
                        type_pred = type_pred.lower()
                        dict_pred['top'] = [type_pred, color_pred, det_clothes_human[i]['top'][:4]]
                print(dict_pred)

                true_top = True if (typ_top in dict_pred['top'][0] and color_top in dict_pred['top'][1]) else False
                true_bottom = True if (typ_bottom in dict_pred['bottom'][0] and color_bottom in dict_pred['bottom'][1]) else False
                if true_top and true_bottom:
                    det_sys.append(det_human[i])
            t10 = time_sync()
            # inference time for efficientNet
            # print(f"Inference time for efficientNet nms time: {t10 - t9:.4f}")
            # -----------------------------------------

        # Tracking object when search suitable
        if len(det_sys):
            det_sys = np.array(det_sys)
            det = torch.from_numpy(det_sys)
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]

            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(),
                                      im0s)  # np array (detections, [xyxy, track_id, class_id])
            # draw boxes for visualization
            # img_numpy = display_video(img_numpy, outputs, color=COLOR[0], search=search)
        else:
            deepsort.increment_ages()

        # save video
        if args.savevid:
            output.write(annotator.im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--yolact_weight', type=str,
                        default="/content/gdrive/MyDrive/model/yolact_plus_resnet50_6_144000.pth")
    parser.add_argument('--yolo_weight', type=str, default="/content/gdrive/MyDrive/model/v5s_human_mosaic.pt")
    parser.add_argument('--type_clothes_weight', type=str, default="/content/gdrive/MyDrive/model/b1_type_clothes.pt")
    parser.add_argument('--color_clothes_weight', type=str, default="/content/gdrive/MyDrive/model/b1_color_clothes.pt")
    parser.add_argument("--cfg_deepsort", type=str, default="/content/HumanRetrieval/deep_sort/deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument('--top', type=str, default=None, help='Torso of human, type and color clothes')
    parser.add_argument('--bottom', type=str, default=None, help='Leg of human, type and color clothes')
    parser.add_argument('--extractor', type=str, default='efficientnet-b0')
    parser.add_argument('--cls_data', type=str, default="EfficientNET/config/dataset.yaml")
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--humans', type=str)
    parser.add_argument('--clothes', type=str, default='short_sleeved_shirt')
    parser.add_argument('--view_img', action="store_true")
    parser.add_argument('--savevid', action="store_true")
    parser.add_argument('--savename', type=str, default="/content/out.avi")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)
