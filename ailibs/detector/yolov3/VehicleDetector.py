import random

import cv2
import torch
import numpy as np

from ailibs.utils.utils import *
from ailibs.detector.yolov3.models import *
from ailibs.utils.datasets import letterbox
from ailibs.utils.iou import non_max_suppression
from ailibs.detector.Detector import OutputPrediction
from ailibs.detector.yolov3.YOLOBaseDetector import YOLOBaseDetector


class VehicleDetector(YOLOBaseDetector):
    """Vehicle detector
    """
    def __init__(self,
                    weights,
                    cfg,
                    names,
                    imgsz=416,
                    conf_thres=0.5,
                    iou_thres=0.6,
                    wh_thres=(30,30),
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
            wh_thres (tuple, optional): width height threshold for ignoring small bounding box. Defaults to (30,30).
            device (str, optional): device. Defaults to "".
            half (bool, optional): half to improve performance. Defaults to False.
        """
        super(VehicleDetector, self).__init__(weights, cfg, names, imgsz, conf_thres, iou_thres, device, half)
        self.__wh_thres = wh_thres # width height threshold for ignoring small vehicle objects.

    def detect(self,
                image,
                plot=False):
        """Detect vehicle in the image

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

        vhc_objs = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    if self._names[int(cls)] in ['car', 'motorcycle', 'bus', 'truck']:
                        xyxy = [x.int().item() for x in xyxy]
                        length_x = xyxy[2] - xyxy[0]
                        length_y = xyxy[3] - xyxy[1]
                        if length_x >= self.__wh_thres[0] and length_y >= self.__wh_thres[1]: # width and height large enough
                            tl = (xyxy[0], xyxy[1])
                            br = (xyxy[2], xyxy[3])
                            vhc_objs.append( OutputPrediction(tl, br, conf.item(), cls.item()) )
                
                if plot:
                    for vhc_obj in vhc_objs:
                        tl, br = vhc_obj.tl, vhc_obj.br
                        cls = vhc_obj.cls
                        cv2.rectangle(image, tl, br, color=self._colors[int(cls)], thickness=2)

        return vhc_objs

    @property
    def wh_thres(self):
        """Get weight height threshold

        Returns:
            tuple: width height threshold
        """
        return self.__wh_thres

    @wh_thres.setter
    def wh_thres(self, wh):
        """Set width height threshold

        Args:
            wh (tuple): width height threshold

        Raises:
            AttributeError: wh must be tuple and contains two integer number.
        """
        if not isinstance(wh, tuple) or len(wh) != 2:
            raise AttributeError("wh must be tuple and contains two integer number.")
        self.__wh_thres = wh
