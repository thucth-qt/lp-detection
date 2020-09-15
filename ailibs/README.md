# Features
1. Vehicle detection
2. License plate detection
# Run detector guideline
## Import detector
```python
# Import vehicle detector
from ailibs.detector.yolov3.VehicleDetector import VehicleDetector
# Import license plate detector
from ailibs.detector.yolov3.LicensePlateDetector import LicensePlateDetector
```

## Define device(cpu or 0 or 0,1)
```python
from ailibs.utils.torch_utils import select_device
device = select_device(device='')
```
## Declare detector instace
```python
# Vehicle detector
vhc_detector = VehicleDetector(weights=<path-to-weights>,
                                cfg=<path-to-config-file>,
                                names=<path-to-names-file>
                                device=device)
# License plate detector
lp_detector = LicensePlateDetector(weights=<path-to-weights>,
                                    cfg=<path-to-config-file>,
                                    names=<path-to-names-file>
                                    device=device)
```
- Weights extention supported: '.pt'

- Model configuration file: format file follow to yolov3 cfg format, can found at [https://github.com/ultralytics/yolov3/tree/master/cfg](https://github.com/ultralytics/yolov3/tree/master/cfg)

- Class name file: contain name for each class, it separated by a newline character. Example showed below:
```txt
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
```

## Detect object in the image
```python
# Read image file
import cv2
image = cv2.imread(<path-to-image>)
# Detect vehicle
vhc_objs = vhc_detector.detect(image, plot=True) # set plot=True to draw bounding box
# Detect license plate
lp_objs = lp_detector.detect(image, plot=True) # set plot=True to draw bounding box
```

## Output for detection tasks
```python
class OutputPrediction:
    """Data class for output prediction
    """
    def __init__(self, tl, br, conf, cls):
        self.tl = tl
        self.br = br
        self.conf = conf
        self.cls = cls

    def __str__(self):
        return str(self.__dict__)
```
Return a list of OutputPrediction objects, the OutputPrediction object contains top left position, bottom right position, confidence score and class index.
