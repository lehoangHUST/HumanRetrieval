import sys

# Adding for yolact : predict class clothes
sys.path.insert(0, '/content/gdrive/MyDrive/Person_retrival_system/Person_retrival_system/yolact')

# Import file to predict
from yolact.data import COLORS
from yolact.utils.augmentations import FastBaseTransform
from yolact.utils import timer
from yolact.layers.output_utils import postprocess, undo_image_transformation

from yolact.data import cfg

import numpy as np
import torch
from collections import defaultdict

# Classes for clothes
_classes_ = [
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


# Predict display
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """

    iou_thresholds = [x / 100 for x in range(50, 100, 5)]
    coco_cats = {}  # Call prep_coco_cats to fill this
    coco_cats_inv = {}
    color_cache = defaultdict(lambda: {})

    # Display in image/list of image/videos.
    display_bboxes = True
    display_masks = False
    display_scores = True
    display_text = True
    display_lincomb = False
    display_fps = False

    # Score threshold
    score_threshold = 0.3

    # Predict class, bbox, seg
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        # Type parameter t is tensor.
        t = postprocess(dets_out, w, h, visualize_lincomb=display_lincomb,
                        crop_masks=False,
                        score_threshold=score_threshold)
        cfg.rescore_bbox = save

    top_k = 1000
    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:top_k]
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]

        """
        Add code to serach clothes.
        Example: I must choose search in image, list of image or videos.
        Have two case:
            1. If image, video have category of search => Bbox, score, may be have segmentation.
            2. If image, video not have category of search => Not thing.
        """

        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    index = []
    if cls is not None:
        for j in range(num_dets_to_consider):
            if _classes_[classes[j]] in cls:
                index.append(j)
    else:
        for j in range(num_dets_to_consider):
            index.append(j)
    num_dets_to_consider = len(index)

    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if num_dets_to_consider == 0:
        return 0, 0

    for j in range(num_dets_to_consider):
        x1, y1, x2, y2 = boxes[index[j], :]
        score = scores[index[j]]
        # Classes for each detection
        _cls_ = int(classes[index[j]])
        mask = masks[index[j]]
        # Tensor
        if j == 0:
            # Init tensor for bounding box
            bounding_box = torch.tensor([[x1, y1, x2, y2, score, _cls_]])
            mask_clothes = torch.tensor(mask).reshape(1, img_numpy.shape[0], img_numpy.shape[1])
        else:
            bb = torch.tensor([x1, y1, x2, y2, score, _cls_])
            m = torch.tensor(mask).reshape(1, img_numpy.shape[0], img_numpy.shape[1])
            bounding_box = torch.vstack((bounding_box, bb))
            mask_clothes = torch.vstack((mask_clothes, m))

    return bounding_box, mask_clothes


# Pred one image.
# Eval clothes
def run_eval_clothes(net_yolact, search_clothes, img_numpy: np.ndarray):
    global cls
    cls = search_clothes
    cfg.mask_proto_debug = False

    frame = torch.from_numpy(img_numpy).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net_yolact(batch)
    # TODO: make prep_display always return Tensor -> easier to work with later
    # TODO 2: replace prep_display with more elegant function
    bbox, mask = prep_display(preds, frame, None, None, undo_transform=False)

    return bbox, mask
