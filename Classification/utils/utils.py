import torch
import os
import numpy as np

import logging
import platform
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

map_size_model = {
    "efficientnet-b0": 224,
    "efficientnet-b1": 240,
    "efficientnet-b2": 260,
    "efficientnet-b3": 300,
    "efficientnet-b4": 380,
    "efficientnet-b5": 456
}


def get_imgsz(model: str):
    assert model in map_size_model.keys(), "model must be one of pre-defined EfficientNet"
    imgsz = map_size_model[model]
    return imgsz

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def convert_categorial(input):
    """ Convert a one-hot input to categorial """
    if isinstance(input, np.ndarray):
        if input.ndim == 1:
            input = np.expand_dims(input, axis=0)
        output = np.argmax(input, axis=1)
        return output
    if isinstance(input, torch.Tensor):
        if input.ndim == 1:
            input = torch.unsqueeze(input, dim=0)
        output = torch.argmax(input, dim=1)
        return output


def convert_output(dict: dict, input):
    types = dict["Type"] # list
    colors = dict["Color"] # list
    # convert output
    type_output = input[0]
    type_output = torch.softmax(type_output, dim=1)
    type_output = torch.argmax(type_output, dim=1).cpu().detach().numpy() # return a numpy array
    color_output = input[1]
    color_output = (color_output > 0.5).type(torch.int)
    color_output = color_output.cpu().detach().numpy()
    type_pred = types[int(type_output)]
    color_pred = []
    for i in range(color_output.shape[1]):
        if color_output[0, i] == 1:
            color = colors[i]
            color_pred.append(color)
    return type_pred, color_pred #string, list


if __name__ == '__main__':
    ar = np.asarray([1, 0, 0, 0])
    print(convert_categorial(ar))
