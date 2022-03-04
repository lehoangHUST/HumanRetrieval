import sys

# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

# Adding for yolact : predict class clothes.
sys.path.insert(0, '/content/gdrive/MyDrive/Person_retrival_system/Person_retrival_system/yolact')
# Adding for yolov5: predict for human.
sys.path.insert(0, '/content/gdrive/MyDrive/Person_retrival_system/Person_retrival_system/yolov5')
# Adding for deepsort: object tracking for multiple-object.
sys.path.insert(0, '/content/gdrive/MyDrive/Person_retrival_system/Person_retrival_system/deep_sort')

# Yolact
from yolact.data import COCODetection, get_label_map, MEANS
from yolact.yolact import Yolact
from yolact.utils.augmentations import BaseTransform, FastBaseTransform, Resize
from yolact.utils.functions import MovingAverage, ProgressBar
from yolact.layers.box_utils import jaccard, center_size, mask_iou
from yolact.utils import timer
from yolact.utils.functions import SavePath
from yolact.layers.output_utils import postprocess, undo_image_transformation

import pycocotools
import eval_clothes

from Detection.yolact.data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
from pathlib import Path

# Yolov5
from yolov5.models.experimental import attempt_load
from yolov5.utils_yolov5.downloads import attempt_download
from yolov5.utils_yolov5.datasets import LoadImages, LoadStreams
from yolov5.utils_yolov5.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from yolov5.utils_yolov5.plots import Annotator, colors
from yolov5.utils_yolov5.torch_utils import load_classifier, select_device, time_sync
import yolov5.detect as detect

# Deepsort
from Detection.deep_sort.deep_sort_pytorch.utils.parser import get_config
from Detection.deep_sort.deep_sort_pytorch.deep_sort import DeepSort

# Draw plot
from draw import *

# Classification
from Classification.modeling.model import Model
from Classification.utils import utils
from Classification.Classification_dict import dict as cls_dict

# File train model with Yolact
train_model_clothes = 'train_models/yolact_base_clothes_1_30000.pth'
# File train model with Yolov5s
train_model_human = 'train_models/yolo5s.pt'
# Config yaml by deepsort
config_ds = 'train_models/deep_sort.yaml'
# deepsort weights 
#weights_ds = 'Detection/train_models/ckpt.t7'

# Classes for clothes
class_clothes = [
  "short_sleeved_shirt",
  "long_sleeved_shirt",    
  "short_sleeved_outwear",
  "long_sleeved_outwear",
  "vest",
  "sling",
  "shorts",
  "trousers",
  "skirt",
  "short_sleeved_dress",
  "long_sleeved_dress",
  "vest_dress",
  "sling_dress"
]

# Classes for human: male or female.
class_human = [
  'male', 'female'
]

# Pre_train model for person retrival system
# Use Yolact or Solo or Mask RCNN to predict clothes in image/images/video
# Use Yolov5 or YoloX to predict human.

def add_args():
  # # Init add argument from command line
  parser = argparse.ArgumentParser(
    description='YOLACT for predict segmentation clothes.')

# Argument
  parser.add_argument('--search_clothes',
                  default=None, type=str,
                  help='Choose category in clothes to appear after predict with yolact.')
  parser.add_argument('--search_human',
                  default=None, type=str,
                  help='Choose category in human to appear after predict with yolov5.')
  parser.add_argument('--image', default=None, type=str,
                  help='A path to image to evaluate on. Passing in a number will use that index.')
  parser.add_argument('--images', default=None, type=str,
                  help='A folder path to list images to evaluate on. Passing in a number will use that index.')
  parser.add_argument('--video', default=None, type=str,
                  help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
  parser.add_argument('--extractor', default='efficientnet-b0', type=str,
                      help='String represent efficientnet extractor')
  parser.add_argument('--cls_weight', default='Classification/weights/effnet_b0_2011.pt', type=str,
                      help='Path to trained weights of classification model')
  args = parser.parse_args()
  return args

# Config Yolact.
def config_Yolact(yolact_weight):
    # Load config from weight
    print("Loading YOLACT" + '-' * 10)
    model_path = SavePath.from_str(yolact_weight)
    config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % config)
    set_cfg(config)

    with torch.no_grad():
      # Temporarily disable to check behavior
      '''# Use cuda
      use_cuda = True
      if use_cuda:
          cudnn.fastest = True
          torch.set_default_tensor_type('torch.cuda.FloatTensor')
      else:
          torch.set_default_tensor_type('torch.FloatTensor')'''

      # Eval for image, images or video
      net = Yolact()
      net.load_weights(yolact_weight)
      net.eval()
      print("Done loading YOLACT" + '-' * 10)
      return net


# ------------------------------------------------------------

# Config with YOLOv5
def config_Yolov5(weights=train_model_human):
  half = False 
  # Initialize

  set_logging()
  device = select_device('')
  half &= device.type != 'cpu'  # half precision only supported on CUDA

  # Load model
  w = weights[0] if isinstance(weights, list) else weights
  classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
  check_suffix(w, suffixes)  # check weights have acceptable suffix
  pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
  stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
  if pt:
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
      model.half()  # to FP16
    if classify:  # second-stage classifier
      modelc = load_classifier(name='resnet50', n=2)  # initialize
      modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

  return model, stride, names


# Config of deep sort
def config_deepsort():
# initialize deepsort
  cfg = get_config()
  cfg.merge_from_file(config_ds)
  #attempt_download(weights_ds, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
  deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                      max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                      max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                      max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                      use_cuda=True)
  return deepsort


# Run each images.
def evalimage(search_human, search_clothes, img_numpy, imgsz_yolov5):
  search = class_human[search_human[0]] + "_" + "_".join(search_clothes)
  with torch.no_grad():
    # Run eval clothes to predict in image.
    # pred_bbox_clothes (batch, 6) -> torch.Tensor
    # pred_bbox_clothes[i] includes (bbox, conf, class)
    pred_bbox_clothes = eval_clothes.run_eval_clothes(net_yolact,
                          search_clothes = search_clothes,
                          img_numpy = img_numpy)

    # Run eval human: male or female to predict in image.
    # pred_bbox_human (batch, 6) -> torch.Tensor
    # pred_bbox_human[i] includes (bbox, conf, class)
    pred_bbox_human = detect.run_eval_human(model = net_yolov5,
                          img_numpy = img_numpy,
                          classes = search_human,
                          imgsz=imgsz_yolov5)
    '''# TODO: Mai
    # Add debugger to inspect pred_bbox_clothes and pred_bbox_human
    annotator = Annotator(img_numpy, line_width=2, example=str("Yo"))
    # draw bbox for human
    for *xyxy, conf, cls in pred_bbox_human:
      annotator.box_label(xyxy, None, color=(255, 255, 255))

    # draw bbox for clothes
    for *xyxy, conf, cls in pred_bbox_clothes:
      annotator.box_label(xyxy, None, color=(255, 255, 0))

    cv2.imshow("Test", img_numpy)
    cv2.waitKey(0)
    # TODO: end'''
    # A list of objects satisfying 2 properties
    list_det = [] # list of torch.Tensor containing bbox of human
    if type(pred_bbox_clothes) == torch.Tensor and type(pred_bbox_human) == torch.Tensor:
      pred_bbox_clothes = pred_bbox_clothes.cpu().numpy()
      pred_bbox_human = pred_bbox_human.cpu().data.numpy()

      # Calculate inters set A and B 
      def inters(bbox_a, bbox_b):
        # determine the coordinates of the intersection rectangle
        x_left = max(bbox_a[0], bbox_b[0])
        y_left = max(bbox_a[1], bbox_b[1])
        x_right = min(bbox_a[2], bbox_b[2])
        y_right = min(bbox_a[3], bbox_b[3])
        if (x_right - x_left)*(y_right - y_left) >= 0:
          return (x_right - x_left)*(y_right - y_left)
        else:
          return 0

      # Count = length of clothes: Draw bbox.
      # Count = not length of clothes: Not Draw bbox.
      for i in range(pred_bbox_human.shape[0]):
        count = 0
        for j in range(pred_bbox_clothes.shape[0]):
          # Calculate area.
          area_j = (pred_bbox_clothes[j][2] - pred_bbox_clothes[j][0])*(pred_bbox_clothes[j][3] - pred_bbox_clothes[j][1])
          area = inters(pred_bbox_human[i, :4], pred_bbox_clothes[j, :4])
          # Conditional
          if area / area_j > 0.7:
            count += 1
    
        if count == len(search_clothes):
          list_det.append(np.array(pred_bbox_human[i, :], dtype=np.float_).tolist())
      
      # If length of list_det not equal 0 -> use deepsort
      if len(list_det) != 0:
        list_det = np.array(list_det) 
        det = torch.from_numpy(list_det)
        xywhs = xyxy2xywh(det[:, 0:4])
        confs = det[:, 4]
        clss = det[:, 5]

        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img_numpy) # np array (detections, [xyxy, track_id, class_id])
        # draw boxes for visualization 
        #img_numpy = display_video(img_numpy, outputs, color=COLOR[0], search=search)
      else:
        deepsort.increment_ages()

    # Classification part
    bboxes = pred_bbox_clothes[:, :4] # np.ndarray
    type_preds = []
    color_preds = []
    for bbox in bboxes:
        roi = img_numpy[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        cls_input = cls_model.preprocess(roi).to(device)
        cls_output = cls_model(cls_input) # list (type, color)
        # type_pred: string
        # color_pred: list(string)
        type_pred, color_pred = utils.convert_output(cls_dict, cls_output)
        type_preds.append(type_pred)
        color_preds.append(color_pred)

    # Draw search: Human + clothes
    cv2.putText(img_numpy, search, (0, 60-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    plt.imshow(img_numpy)
    plt.title(str(type_preds) + "\n" + str(color_preds))
    plt.show()
    return img_numpy

# Run list image in folder
"""
    --folder_images/
      --1.jpg
      --2.jpg
      --3.jpg
      ......
"""
def evalimages(inp, out, search_human, search_clothes, imgsz):
  # Check exist path folder
  if not os.path.exists(out):
    os.mkdir(out)

  count = 0
  # Pred for each image in folder image
  for image in os.listdir(inp):
    count += 1
    img_pred = evalimage(search_human, search_clothes, cv2.imread(inp + '/' + image), imgsz)
    cv2.imwrite(out + str(count) + '.jpg', img_pred)

# Run video to predict
def evalvideo(inp, out, search_human, search_clothes, imgsz):
  # Inputs
  cap = cv2.VideoCapture(inp)

  # Open video to read
  if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(width, height)
    res=(int(width), int(height))
    # this format fail to play in Chrome/Win10/Colab
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
    # fourcc = cv2.VideoWriter_fourcc(*'H264') #codec
    output = cv2.VideoWriter(out, fourcc, 30, res)

    frame = None
    while True:
        try:
            is_success, frame = cap.read()
        except cv2.error:
            continue

        if not is_success:
            break

        # OPTIONAL: do some processing
        # convert cv2 BGR format to RGB
        # Path image or image: to run one image/ list image or videos.
        img_numpy = evalimage(search_human, search_clothes, frame, imgsz)

        output.write(img_numpy)
    # OPTIONAL: show last image
  cap.release()


"""
    Main run 3 function:
      1. Predict for human use Model YOLOV5
      2. Predict for category clothes use Model YOLACT/YOLACT++/SOLOv1/SOLOv2.
      3. After Predict clothes and human => Matching requires of problem.

"""
# Run video : Input format is <path_video_input>.mp4:<path_video_output>.mp4
# <path_video_input>.mp4 => Video to processing human, clothes and color of clothes.
# <path_video_output>.mp4 => Save video after processing video input

def run(args):
  # Search clothes.
  half = False
  # Initialize
  global device
  device = select_device('cuda')
  half &= device.type != 'cpu'  # half precision only supported on CUDA

  if ',' in args.search_clothes:
    search_clothes = args.search_clothes.split(',')
  else:
    search_clothes = [args.search_clothes]
  
  if not all(elem in class_clothes for elem in search_clothes):
    raise ValueError(f"Have any category not exist in parameter classes")
  
  # Config yolact
  global net_yolact
  net_yolact = config_Yolact(train_model_clothes).cuda()


  # Config yolov5
  global net_yolov5
  # Load model
  imgsz = 640
  net_yolov5 = attempt_load(train_model_human, map_location=device)  # load FP32 model
  stride = int(net_yolov5.stride.max())  # model stride
  imgsz = check_img_size(imgsz, s=stride)  # check img_size
  names = net_yolov5.module.names if hasattr(net_yolov5, 'module') else net_yolov5.names  # get class names

  # Config deepsort
  global deepsort
  deepsort = config_deepsort()

  # Config classification model
  # --------------------------------------------
  global cls_model
  cls_model = Model(args.extractor,
                    True,
                    len(cls_dict["Type"]),
                    len(cls_dict["Color"]))
  cls_model.load_state_dict(torch.load(args.cls_weight))
  cls_model.eval()
  # --------------------------------------------

  if half:
    net_yolov5.half()  # to FP16

  search_human = args.search_human.split(',')
  # Process data input: Example 
  if len(search_human) == 1:
    search_human = [class_human.index("".join(search_human))]
  else:
    search_human = [0, 1] # With 0 is male and 1 is female.

  # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages, and evalvideo
  if args.image is not None:
    inp, out = args.image.split(':')
    img_pred = evalimage(search_human, search_clothes, cv2.imread(inp), imgsz)
    # Save file img pred by clothes and human.
    cv2.imwrite(out, img_pred)
    return
  elif args.images is not None:
    inp, out = args.images.split(':')
    evalimages(inp, out, search_human, search_clothes, imgsz)
    return
  elif args.video is not None:
    inp, out = args.video.split(':')
    evalvideo(inp, out, search_human, search_clothes, imgsz)
    return


if __name__ == '__main__':
  args = add_args()
  run(args)