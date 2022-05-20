# outside lib import
import argparse
import numpy as np
import yaml

import torch
import torch.backends.cudnn as cudnn

# own lib import
import modules
from Classification.modeling.model_v2 import Model_type
from Classification.modeling.model_v2 import Model_color
from Detection.yolov5.utils.datasets import LoadImages, LoadStreams, IMG_FORMATS
from Detection.yolov5.utils.torch_utils import time_sync, select_device
from Detection.yolov5.utils.general import set_logging, non_max_suppression, scale_coords
from Detection.yolov5.utils.plots import Annotator

from Detection.eval_clothes import run_eval_clothes

from Classification.utils import utils

from utils.metrics import accuracy_system, split_label

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
    "sling",
    "vest"
]

bottom = [
    "short",
    "trousers",
    "skirt",
    "vest_dress"
]


def remove_clothes(yolact_preds_bbox: torch.Tensor, yolact_preds_mask: torch.Tensor):
    yolact_preds_type = list(map(int, yolact_preds_bbox[:, 5].tolist()))
    bbox = torch.zeros(2, 4)
    typ = torch.zeros(2, 1)
    mask = torch.zeros(2, yolact_preds_mask.shape[1], yolact_preds_mask.shape[2])
    index, up_body, down_body = 0, False, False

    for i, pred_type in enumerate(yolact_preds_type):
        if classes[pred_type] in top and not up_body:
            up_body = True
            bbox[index, :] = yolact_preds_bbox[i, :4]
            typ[index, :] = yolact_preds_bbox[i, 5]
            mask[index, :] = yolact_preds_mask[i, :]
            index += 1
        if classes[pred_type] in bottom and not down_body:
            down_body = True
            bbox[index, :] = yolact_preds_bbox[i, :4]
            typ[index, :] = yolact_preds_bbox[i, 5]
            mask[index, :] = yolact_preds_mask[i, :]
            index += 1

        if up_body and down_body:
            break

    return bbox, mask, typ


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
    # disable for now
    '''if not all(elem in class_clothes for elem in search_clothes):
        raise ValueError(f"Have any category not exist in parameter classes")'''

    # Load nets
    net_YOLACT, yolact_name = modules.config_Yolact(args.yolact_weight)
    net_YOLO, strides, yolo_name, imgsz = modules.config_Yolov5(args.yolo_weight, device)

    # Type clothes 
    net_type = Model_type(args.extractor,
                          use_pretrained=False,
                          num_class=len(cls_dataset['class']['Type']))
    net_type.load_state_dict(torch.load(args.type_clothes_weight)['state_dict'])
    net_type.to(device)
    net_type.eval()

    # Color clothes 
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

    # Preds label: Human + Type clothes + Color clothes
    # True label : Path file include Human + Type clothes + Color clothes
    true_label = []
    pred_label = []
    # Run Inference
    for index, (path, im, im0s, vid_cap, _) in enumerate(dataset):
        print(path)
        label = ""
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
        # Inference on yolov5
        yolo_preds = net_YOLO(im_yolo)  # (batch, (bbox, conf, class)) type torch.Tensor
        t3 = time_sync()
        # nms for yolo
        # yolo_preds: torch.Tensor
        yolo_preds = non_max_suppression(yolo_preds, 0, 0.4, None, max_det=100)[0]
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
        yolact_preds_bbox, yolact_preds_mask = run_eval_clothes(net_YOLACT,
                                                                search_clothes=None,
                                                                img_numpy=im_yolact)

        # Remove clothes not suitable
        yolact_preds_bbox, yolact_preds_mask, typ = remove_clothes(yolact_preds_bbox, yolact_preds_mask)

        if not isinstance(yolact_preds_bbox, int):

            # Convert torch float to torch int
            bbox = yolact_preds_bbox.type(torch.int32).cpu().numpy()
            yolact_preds_mask = yolact_preds_mask.type(torch.uint8).cpu().numpy()
            typ = typ.type(torch.uint8).cpu().numpy()

            # changebackground by color
            # =====================================================
            mask_clothes = []
            for i, mask in enumerate(yolact_preds_mask):
                img = im0s.copy()
                for j in range(bbox[i][1], bbox[i][3]):
                    for k in range(bbox[i][0], bbox[i][2]):
                        if mask[j, k] == 0:
                            img[j, k, :] = 255
                mask_clothes.append(img[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2], :])

            # =====================================================

            # classification
            # ======================================================
            # 1. Read every single clothes ROI from yolact output one by one
            # 2. Perform preprocess to ROI
            # 3. Perform forward pass on image
            # 4. Convert output from classification model to correct read-able format
            # 5. Draw bbox with type and color label
            # =======================================================
            dict_pred = {}
            for idx, mask in enumerate(mask_clothes):
                inp = net_type.preprocess(mask)
                color_output = net_color(inp)
                if classes[typ[idx][0]] in bottom:
                    colors = cls_dataset['class']['Color']
                    color_pred = []
                    color_output = (color_output > 0.5).type(torch.int)
                    color_output = color_output.cpu().detach().numpy()
                    for i in range(color_output.shape[1]):
                        if color_output[0, i] == 1:
                            color = colors[i]
                            color_pred.append(color)
                    dict_pred['bottom'] = [classes[typ[idx][0]], color_pred]

                if classes[typ[idx][0]] in top:
                    clothes_output = net_type(inp)
                    # type_pred: string
                    # color_pred: list(string)
                    type_pred, color_pred = utils.convert_output(cls_dataset['class'], [clothes_output, color_output])
                    type_pred = type_pred.lower()
                    dict_pred['top'] = [type_pred, color_pred]
            dict_pred['gender'] = yolo_name[int(yolo_preds[0, 5])]

            # Append pred_label and true_label
            pred_label.append(dict_pred)
            true_label.append(split_label(path))

    accuracy_system(pred_label, true_label)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='')
    parser.add_argument('--yolact_weight', type=str,
                        default="/content/gdrive/MyDrive/model/yolact_plus_resnet50_6_144000.pth")
    parser.add_argument('--yolo_weight', type=str, default="/content/gdrive/MyDrive/model/best.pt")
    parser.add_argument('--type_clothes_weight', type=str,
                        default="/content/gdrive/MyDrive/model/EffNet_B2_type_Aug/efficientnet-b2type_clothes.pt")
    parser.add_argument('--color_clothes_weight', type=str,
                        default="/content/gdrive/MyDrive/model/EffNet_B2_color_Aug/efficientnet-b2color_clothes.pt")
    parser.add_argument('--extractor', type=str, default='efficientnet-b0')
    parser.add_argument('--cls_data', type=str,
                        default="/content/gdrive/MyDrive/HumanRetrieval_v2/Classification/config/dataset.yaml")
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--humans', type=str)
    parser.add_argument('--clothes', type=str, default='short_sleeved_shirt')
    parser.add_argument('--view_img', action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)
