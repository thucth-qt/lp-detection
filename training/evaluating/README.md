# Evaluating model
## Installation
1. Setup environment for evaluating model
```bash
# Create virtual environment
virtualvenv -p python3 <name-venv>
# Activate environment
source <name-venv>/bin/activate
# Install dependencies
pip install -r requirements.txt
```
2. Preparing dataset
- Annotation dataset follow YOLO format
class index, x center, y center, width, height
```txt
0, 0.3123, 0.3132, 0.123, 0.3141
```
- Add image paths to csv file as below
```
ailibs_data/dataset/video_traffic_Lane-splitting+in+Tokyo.17753_1.png
ailibs_data/dataset/video_traffic_Lane-splitting+in+Tokyo.24387_0.png
ailibs_data/dataset/video_parking_lot_In+Japanese+car+park,+spot+parks+YOU!!258_0.png
ailibs_data/dataset/video_motorbike_motorcycle+paradise+Japan10526_1.png
ailibs_data/dataset/video_container_Export+of+++4x4+Japanese+Mini+Truck+Container++from+Japan.970_0.png
```
- Change valid csv file into data file(lp.data)
```
classes=1
train=<train-file-path>
valid=<valid-file-path>
test=<test-file-path>
names=ailibs/detector/yolov3/cfg/lp.names
```
3. Run evaluating with command
```bash
python training/evaluating/evaluate.py
```

## Evaluation metric
- Precision
- Recall
- F1-score
- mAP