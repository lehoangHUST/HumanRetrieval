import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

"""
    Inputs: Video have suffixes .mp4 ,.....
    Output: List img in Video save folder.
"""

VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv'] 

# Argument to implement read Video
def add_args():
  # Init Argument
  parser = argparse.ArgumentParser()

  # Add Argument
  parser.add_argument('--inputs', type = str,
                      default=None,
                      help='Inputs is video, not img.')
  parser.add_argument('--outputs', type = str,
                      default=None,
                      help='One folder contains image after read video in inputs.')
  
  args = parser.parse_args()
  return args

# Main
def run(args):

  # Check file inputs exist.
  assert os.path.isfile(args.inputs), f"Not found file in system."
  # Make dir file ouputs
  cap = cv2.VideoCapture(args.inputs)

  if cap.isOpened():
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(width, height)
    res=(int(width), int(height))
    # this format fail to play in Chrome/Win10/Colab
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #codec
    # fourcc = cv2.VideoWriter_fourcc(*'H264') #codec

    count = 0
    frame = None
    while True:
        try:
            is_success, frame = cap.read()
        except cv2.error:
            continue

        if not is_success:
            break

        count += 1
        # Save img in folder path
        cv2.imwrite(args.outputs + '/' + str(count) + '.jpg', frame)


    print(count)
  cap.release()


if __name__ == '__main__':
  args = add_args()
  run(args)
