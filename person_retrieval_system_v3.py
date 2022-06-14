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
from Detection.yolor.utils.datasets import LoadImages, LoadStreams, img_formats
from Detection.yolor.utils.torch_utils import select_device, load_classifier, time_synchronized
from Detection.yolor.utils.general import set_logging, non_max_suppression, xyxy2xywh, scale_coords

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

humans = [
    "male",
    "female"
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

    # Load nets YOLOr
    net_YOLOR, imgsz = modules.config_Yolor(args.cfg_yolor, args.yolor_weight, device)

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
                             auto_size=64)  # (path, letterbox_img: np, orig_img: cv2, cap)

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
    for index, (path, im, im0s, vid_cap) in enumerate(dataset):
        det_sys = []
        human_label = ""
        is_img = True if any(ext in path for ext in img_formats) else False

        # yolor inference
        # -----------------------------------------
        im_yolor = torch.from_numpy(im).to(device)  # yolo input
        im_yolor = im_yolor.float()
        im_yolor /= 255
        if len(im_yolor.shape) == 3:
            im_yolor = im_yolor[None]  # expand for batch dim
      
        t1 = time_synchronized()
        yolor_preds = net_YOLOR(im_yolor)[0]  # (batch, (bbox, conf, class)) type torch.Tensor
        
        yolor_preds = non_max_suppression(yolor_preds, 0.6, 0.5, humans, None)[0]
      
        if len(yolor_preds):
            yolor_preds[:, :4] = scale_coords(im_yolor.shape[2:], yolor_preds[:, :4], im0s.shape).round()
        t2 = time_synchronized()
        print(f"Inference time for YOLOR: {t2 - t1:.4f}")
        
        # -----------------------------------------

        # yolact inference
        # -----------------------------------------
        # TODO: re-write the run_eval_clothes function, drop FastBaseTransform, drop prep_display
        im_yolact = im0s.copy()  # copy to another image so we can draw on im0s later
        # type torch.Tensor, shape (batch, (bbox, conf, cls))
        # type int if no detection
        
        t3 = time_synchronized()
        yolact_preds_bbox, yolact_preds_mask = run_eval_clothes(net_YOLACT,
                                                                search_clothes=clothes,
                                                                img_numpy=im_yolact)
        t4 = time_synchronized()
       
        # inference time for YOLACT
        print(f"Inference time for YOLACT: {t4 - t3:.4f}")

        if not isinstance(yolact_preds_bbox, int):
            # Matching yolact & yolov5
            # =====================================================
            
            t5 = time_synchronized()

            det_human, det_clothes_human = matching(yolor_preds, yolact_preds_bbox, clothes)
            
            t6 = time_synchronized()

            print(f"Inference time for Matching human and clothes: {t6 - t5:.4f}")

            # inference time for matching yolact & yolov5
            # print(f"Inference time for YOLACT nms time: {t8 - t7:.4f}")
            # =====================================================

            # change background by color
            # =====================================================
            t7 = time_synchronized()
            mask_det = []
            for det_cls in det_clothes_human:
                mask_clothes = {}
                for k, (body, bbox) in enumerate(det_cls.items()):
                    img = im0s[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    mask = yolact_preds_mask[k, bbox[1]:bbox[3], bbox[0]:bbox[2]].type(torch.uint8).cpu().numpy()
                    masks = np.array([mask, mask, mask]).transpose(1, 2, 0)
                    masks_white = (1 - masks) * 255
                    img *= masks
                    img += masks_white
                    cv2.imwrite('/content/' + str(k+1) + '.jpg', img)
                    
                    if body == 'top':
                        mask_clothes['top'] = img
                    elif body == 'bottom':
                        mask_clothes['bottom'] = img
          
                mask_det.append(mask_clothes)
            t8 = time_synchronized()
            print(f"Inference time for change background: {t8 - t7:.4f}")



            # ======================================================
            # classification
            # ======================================================
            # 1. Read every single clothes ROI from yolact output one by one
            # 2. Perform preprocess to ROI
            # 3. Perform forward pass on image
            # 4. Convert output from classification model to correct read-able format
            # 5. Draw bbox with type and color label
            # =======================================================
            t9 = time_synchronized()
            clothes_labels = []
            for i, mask_clothes in enumerate(mask_det):
                dict_pred = {}
                for body, mask in mask_clothes.items():
                    inp = net_type.preprocess(mask)
                    if body == 'bottom':
                        t13 = time_synchronized()
                        color_output = net_color(inp)
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
                        t14 = time_synchronized()
                        print(f"Time inference EfficientNet part bottom: {t14 - t13:.4f}")

                    if body == 'top':
                        t11 = time_synchronized()
                        clothes_output = net_type(inp)
                        color_output = net_color(inp)
                        # type_pred: string
                        # color_pred: list(string)
                        type_pred, color_pred = utils.convert_output(cls_dataset['class'],
                                                                     [clothes_output, color_output])
                        type_pred = type_pred.lower()
                        dict_pred['top'] = [type_pred, color_pred, det_clothes_human[i]['top'][:4]]
                        t12 = time_synchronized()
                        print(f"Time inference EfficientNet part top: {t12 - t11:.4f}")
                print(dict_pred)

                true_top = True if (typ_top in dict_pred['top'][0] and color_top in dict_pred['top'][1]) else False
                true_bottom = True if (typ_bottom in dict_pred['bottom'][0] and color_bottom in dict_pred['bottom'][1]) else False
                if true_top and true_bottom:
                    det_sys.append(det_human[i])
            t10 = time_synchronized()
            # inference time for efficientNet
            print(f"Inference time for efficientNet nms time: {t10 - t9:.4f}")
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
    parser.add_argument('--cfg_yolor', type=str, default='/content/gdrive/MyDrive/HumanRetrieval_v3/Detection/yolor/cfg/yolor_csp.cfg')
    parser.add_argument('--yolor_weight', nargs='+', type=str, default='/content/best.pt', help='model.pt path(s)')
    parser.add_argument('--type_clothes_weight', type=str, default="/content/gdrive/MyDrive/model/b1_type_clothes.pt")
    parser.add_argument('--color_clothes_weight', type=str, default="/content/gdrive/MyDrive/model/b1_color_clothes.pt")
    parser.add_argument("--cfg_deepsort", type=str, default="/content/gdrive/MyDrive/HumanRetrieval_v3/Detection/deep_sort/configs/deep_sort.yaml")
    parser.add_argument('--top', type=str, default=None, help='Torso of human, type and color clothes')
    parser.add_argument('--bottom', type=str, default=None, help='Leg of human, type and color clothes')
    parser.add_argument('--extractor', type=str, default='efficientnet-b0')
    parser.add_argument('--cls_data', type=str, default="EfficientNET/config/dataset.yaml")
    parser.add_argument('--source', type=str, default='0')
    parser.add_argument('--humans', type=str, default=None)
    parser.add_argument('--clothes', type=str, default='short_sleeved_shirt')
    parser.add_argument('--view_img', action="store_true")
    parser.add_argument('--savevid', action="store_true")
    parser.add_argument('--savename', type=str, default="/content/out.avi")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)
