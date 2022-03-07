# general import
import torch
import torch.backends.cudnn as cudnn

import os, sys
sys.path.insert(0, "Detection")


# YOLACT
# ------------------------------------------------------------
from yolact.yolact import Yolact
from yolact.data import set_cfg
from yolact.utils.functions import SavePath


# TODO: make config as a parameter instead of using a global parameter from yolact.data
def config_Yolact(yolact_weight):
    # Load config from weight
    print("Loading YOLACT" + '-'*10)
    model_path = SavePath.from_str(yolact_weight)
    config = model_path.model_name + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % config)
    cfg = set_cfg(config)

    names = cfg.dataset.class_names

    with torch.no_grad():
        # Temporarily disable to check behavior
        # Behavior: disabling this cause torch.Tensor(list, device='cuda') not working
        # Currently enable for now
        # TODO: Will find a workaround to disable this behavior
        # Use cuda
        use_cuda = True
        if use_cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Eval for image, images or video
        net = Yolact()
        net.load_weights(yolact_weight)
        net.eval()
        print("Done loading YOLACT" + '-'*10)
        return net.cuda(), names
# ------------------------------------------------------------



# YOLOv5
# ------------------------------------------------------------
sys.path.insert(0, "Detection/yolov5")

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size

def config_Yolov5(yolo_weight, device, imgsz=640):
    # Load model
    model = DetectMultiBackend(yolo_weight, device=device)  # load FP32 model
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    return model, stride, names, imgsz

# ------------------------------------------------------------
# ------------------------------------------------------------



# Classification
# ------------------------------------------------------------
# TODO: create a yaml config file to load classification model from
sys.path.insert(0, "Classification")
from Classification.modeling.model import Model
from Classification.modeling.model_v2 import Model_type
from Classification.modeling.model_v2 import Model_color

# Model type + color
def config_clsmodel(weight, base_extractor, num_cls1, num_cls2, device):
    model = Model(base_model=base_extractor,
                  use_pretrained=False,
                  num_class_1=num_cls1,
                  num_class_2=num_cls2)
    model.load_state_dict(torch.load(weight)['state_dict'])
    model.eval()
    model.to(device)
    return model

# Model type
def config_typemodel(weight, base_extractor, num_cls, device):
    model = Model_type(base_model=base_extractor,
                  use_pretrained=False,
                  num_class=num_cls)
    model.load_state_dict(torch.load(weight)['state_dict'])
    model.eval()
    model.to(device)
    return model

# Model color
def config_colormodel(weight, base_extractor, num_cls, device):
    model = Model_color(base_model=base_extractor,
                  use_pretrained=False,
                  num_class=num_cls)
    model.load_state_dict(torch.load(weight)['state_dict'])
    model.eval()
    model.to(device)
    return model

# ------------------------------------------------------------