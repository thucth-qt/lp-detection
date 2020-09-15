import random

import cv2
import torch
import numpy as np

from ailibs.utils.utils import *
from ailibs.detector.yolov3.models import *
from ailibs.utils.datasets import letterbox
from ailibs.utils.iou import non_max_suppression
from ailibs.detector.Detector import Detector, OutputPrediction


class YOLOBaseDetector(Detector):
    """YOLO base detector
    """
    def __init__(self,
                    weights,
                    cfg,
                    names,
                    imgsz=416,
                    conf_thres=0.5,
                    iou_thres=0.6,
                    device="",
                    half=False):
        """Initialize parameters

        Args:
            weights (str): weights path for vehicle detector
            cfg (str): configure file
            names (str or list): names for classes
            imgsz (int, optional): image size to inference. Defaults to 416.
            conf_thres (float, optional): confidence threshold. Defaults to 0.5.
            iou_thres (float, optional): Iou threshold. Defaults to 0.6.
            device (str, optional): device. Defaults to "".
            half (bool, optional): half to improve performance. Defaults to False.
        """
        # Check config file exist
        cfg = check_file(cfg)

        # Initial model
        self._model = Darknet(cfg, imgsz)

        self._imgsz = imgsz
        self._device = select_device(device=device) if not device else device

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            self._model.load_state_dict(torch.load(weights, map_location=self._device)['model'])
        else:  # darknet format
            load_darknet_weights(self._model, weights)

        # Evaluation mode
        self._model.to(self._device).eval()

        # Half precision to improve performance
        self._half = half and self._device.type != 'cpu'  # half precision only supported on CUDA
        if self._half:
            self._model.half()

        # Create colors for classes
        self._names = names if isinstance(names, list) else load_classes(names)
        self._colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        # Threshold
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres

    def detect(self,
                image,
                plot=False):
        """Detect license plate in the image

        Args:
            image (numpy.ndarray): input image 
            plot (bool, optional): drawing detected vehicle box. Defaults to False.

        Returns:
            list: list of OutputPrediction objects
        """
        # Padded resize
        img = letterbox(image, new_shape=self._imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Convert numpy to tensor
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(self._device) # to device
        img = img.half() if self._half else img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Prediction
        pred = self._model(img)[0]

        # to float
        if self._half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, self._conf_thres, self._iou_thres)

        lp_objs = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    xyxy = [x.int().item() for x in xyxy]
                    tl = (xyxy[0], xyxy[1])
                    br = (xyxy[2], xyxy[3])
                    lp_objs.append( OutputPrediction(tl, br, conf.item(), cls.item()) )
                
                if plot:
                    for lp_obj in lp_objs:
                        tl, br = lp_obj.tl, lp_obj.br
                        cls = lp_obj.cls
                        cv2.rectangle(image, tl, br, color=self._colors[int(cls)], thickness=2)

        return lp_objs

    @property
    def conf_thres(self):
        """Get confidence threshold

        Returns:
            float: confidence threshold
        """
        return self._conf_thres

    @conf_thres.setter
    def conf_thres(self, conf):
        """Set confidence threshold

        Args:
            conf (float): confidence threshold

        Raises:
            FloatingPointError: Confidence threshold in range from 0 to 1.
        """
        if not 0 <= conf <= 1:
            raise FloatingPointError("Confidence threshold in range from 0 to 1.")
        self._conf_thres = conf

    @property
    def iou_thres(self):
        """Get Iou threshold

        Returns:
            float: Iou threshold
        """
        return self._iou_thres

    @iou_thres.setter
    def iou_thres(self, iou):
        """Set iou threshold

        Args:
            iou (float): Iou threshold

        Raises:
            FloatingPointError: Iou threshold in range from 0 to 1.
        """
        if not 0 <= iou <= 1:
            raise FloatingPointError("Iou threshold in range from 0 to 1.")
        self._iou_thres = iou

    @property
    def device(self):
        """Get device

        Returns:
            str: cpu or 0 or 0,1 ...
        """
        return self._device

    @device.setter
    def device(self, device):
        """Set other device

        Args:
            device (str): device

        Raises:
            AttributeError: Device should be cpu or 0 or 0,1 ...
        """
        if not isinstance(device, str):
            raise AttributeError("Device should be cpu or 0 or 0,1 ...")
        self._device = select_device(device=device)
        self._model.to(self._device)
        self._half = self._half and self._device.type != 'cpu'

    @property
    def names(self):
        """Get all class names

        Returns:
            list: class names
        """
        return self._names