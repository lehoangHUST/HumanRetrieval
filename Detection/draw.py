"""
    - Display for object tracking: human + clothes in video.
    - The following tasks need to be performed:
      1. Draw bbox with color.
      2. Draw text for bbox of object
      3. Draw text to display FPS in video
"""
from numpy.random import randint
import torch
import cv2

COLOR = [
  (255, 0, 0), # Red
  (0, 255, 0), # Green
  (0, 0, 255), # Blue
  (255, 255, 255) # White
]


# Draw in video
def display_video(img, inp, color: tuple, search: str):
  if isinstance(inp, torch.Tensor):
    inp = torch.from_numpy(inp)
  
  # Draw det
  for det in inp:
    x1y1 = (det[0], det[1])
    x2y2 = (det[2], det[3])

    # Draw bbox of object
    img_numpy = cv2.rectangle(img, x1y1, x2y2, color, thickness=1)

    # Draw text for object
    cv2.putText(img_numpy, 'Object' + str(det[4]), (det[0], det[1]-20), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    return img_numpy
  