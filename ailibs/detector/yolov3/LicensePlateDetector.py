import random

import torch
import numpy as np

from ailibs.utils.utils import *
from ailibs.detector.yolov3.models import *
from ailibs.utils.datasets import letterbox
from ailibs.utils.iou import non_max_suppression
from ailibs.detector.Detector import OutputPrediction
from ailibs.detector.yolov3.YOLOBaseDetector import YOLOBaseDetector


class LicensePlateDetector(YOLOBaseDetector):
    
    def __init__(self,
                    weights,
                    cfg,
                    names,
                    imgsz=320,
                    conf_thres=0.5,
                    iou_thres=0.6,
                    device="",
                    half=False):
        """Initialize parameters

        Args:
            weights (str): weights path for vehicle detector
            cfg (str): configure file
            names (str or list): names for classes
            imgsz (int, optional): image size to inference. Defaults to 320.
            conf_thres (float, optional): confidence threshold. Defaults to 0.5.
            iou_thres (float, optional): Iou threshold. Defaults to 0.6.
            device (str, optional): device. Defaults to "".
            half (bool, optional): half to improve performance. Defaults to False.
        """
        super(LicensePlateDetector, self).__init__(weights, cfg, names, imgsz, conf_thres, iou_thres, device, half)
