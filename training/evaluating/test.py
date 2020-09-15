import os
import sys
import unittest
PYTHON_PATH = os.path.abspath('./')
sys.path.insert(0, PYTHON_PATH)

import numpy as np
import torch

from ailibs.utils.torch_utils import select_device
from training.evaluating.evaluate import load_model
from training.metrics import ap_per_class


class TestSelectDevice(unittest.TestCase):
    """Test case for select device cpu or 0 or 0,1
    """
    def test_cpu(self):
        """Test cpu device
        """
        input_ = 'cpu'
        output = select_device(input_)
        expect = torch.device('cpu')
        self.assertEqual(output, expect)

    def test_gpu0(self):
        """Test GPU device 0
        """
        input_ = '0'
        output = select_device(input_)
        expect = torch.device('cuda:0')
        self.assertEqual(output, expect)


class TestModelLoader(unittest.TestCase):
    """Testcase for model loader
    """
    def test_num_of_modules(self):
        """Test get number of modules
        """
        cfg = os.path.join('ailibs', 'detector', 'yolov3', 'cfg', 'yolov3-lp.cfg')
        weights = os.path.join('ailibs_data', 'weights', 'lp.pt')
        imgsz = 320
        device = torch.device('cpu')
        model = load_model(cfg, imgsz, weights, device)
        self.assertEqual(len(model.module_list), 107)

    def test_get_yolo_layers(self):
        """Test get yolo layers indices
        """
        cfg = os.path.join('ailibs', 'detector', 'yolov3', 'cfg', 'yolov3-lp.cfg')
        weights = os.path.join('ailibs_data', 'weights', 'lp.pt')
        imgsz = 320
        device = torch.device('cpu')
        model = load_model(cfg, imgsz, weights, device)
        expect = [82, 94, 106]
        self.assertEqual(model.yolo_layers, expect)

    def test_num_of_layers_fuse(self):
        """Test get number of layers when fuse model
        """
        cfg = os.path.join('ailibs', 'detector', 'yolov3', 'cfg', 'yolov3-lp.cfg')
        weights = os.path.join('ailibs_data', 'weights', 'lp.pt')
        imgsz = 320
        device = torch.device('cpu')
        model = load_model(cfg, imgsz, weights, device, fuse=True)
        self.assertEqual(len(list(model.parameters())), 150)

    def test_num_of_layers_nonfuse(self):
        """Test get number of layers when non fuse model
        """
        cfg = os.path.join('ailibs', 'detector', 'yolov3', 'cfg', 'yolov3-lp.cfg')
        weights = os.path.join('ailibs_data', 'weights', 'lp.pt')
        imgsz = 320
        device = torch.device('cpu')
        model = load_model(cfg, imgsz, weights, device, fuse=False)
        self.assertEqual(len(list(model.parameters())), 222)

    def test_anchors(self):
        """Test get size anchors
        """
        cfg = os.path.join('ailibs', 'detector', 'yolov3', 'cfg', 'yolov3-lp.cfg')
        weights = os.path.join('ailibs_data', 'weights', 'lp.pt')
        imgsz = 320
        device = torch.device('cpu')
        model = load_model(cfg, imgsz, weights, device)

        anchors = None
        for yolo_layer in model.yolo_layers[::-1]:
            if anchors is None:
                anchors = model.module_list[yolo_layer].anchors
            else:
                anchors = np.concatenate([anchors, model.module_list[yolo_layer].anchors], axis=0)

        anchors_expect = np.array([27, 28, 66, 25,
                            47, 42, 39, 62,
                            71, 37, 64, 54,
                            111, 43, 86, 90,
                            169, 62], dtype=np.float).reshape(-1, 2)

        self.assertTrue((anchors == anchors_expect).all())


class TestMetric(unittest.TestCase):
    """Testcase for evalutation metrics
    """
    def test_precision_recall_single_class(self):
        """Test precision and recall for single class
        """
        correct = np.array([[True], [True]])
        conf = np.array([0.685, 0.702])
        pred_cls = np.array([0., 0.])
        target_cls = np.array([0., 0.])
        p, r, _, _, _ = ap_per_class(correct, conf, pred_cls, target_cls)
        output = np.concatenate([p, r])
        expect = np.array([[1.], [1.]])
        self.assertTrue((output == expect).all())

    def test_precision_recall_two_class(self):
        """Test precision and recall for two class
        """
        correct = np.array([[False], [True], [False]])
        conf = np.array([0.654, 0.702, 0.432])
        pred_cls = np.array([0., 0., 1.])
        target_cls = np.array([1. , 0., 0.])
        p, r, _, _, _ = ap_per_class(correct, conf, pred_cls, target_cls)
        expect_p = np.array([[0.5], [0.]])
        expect_r = np.array([[0.5], [0.]])
        self.assertTrue((p == expect_p).all() and (r == expect_r).all())

    def test_precision_recall_three_class(self):
        """Test precision and recall for three class
        """
        correct = np.array([[False], [True], [False]])
        conf = np.array([0.654, 0.702, 0.432])
        pred_cls = np.array([0., 2., 1.])
        target_cls = np.array([1. , 2., 0.])
        p, r, _, _, _ = ap_per_class(correct, conf, pred_cls, target_cls)
        expect_p = np.array([[0.], [0.], [1.]])
        expect_r = np.array([[0.], [0.], [1.]])
        self.assertTrue((p == expect_p).all() and (r == expect_r).all())

    def test_ap_single_class(self):
        """Test average precision for single class
        """
        correct = np.array([[True], [True]])
        conf = np.array([0.685, 0.702])
        pred_cls = np.array([0., 0.])
        target_cls = np.array([0., 0.])
        _, _, ap, _, _ = ap_per_class(correct, conf, pred_cls, target_cls)
        expect = np.array([[0.995]])
        self.assertTrue((ap == expect).all())

    def test_ap_two_class(self):
        """Test average precision for two class
        """
        correct = np.array([[False], [True], [False]])
        conf = np.array([0.654, 0.702, 0.432])
        pred_cls = np.array([0., 0., 1.])
        target_cls = np.array([1. , 0., 0.])
        _, _, ap, _, _ = ap_per_class(correct, conf, pred_cls, target_cls)
        expect = np.array([[0.5], [0.]])
        self.assertTrue((ap == expect).all())

    def test_f1_single_class(self):
        """Test F1-score for single class
        """
        correct = np.array([[True], [True]])
        conf = np.array([0.685, 0.702])
        pred_cls = np.array([0., 0.])
        target_cls = np.array([0., 0.])
        _, _, _, f1, _ = ap_per_class(correct, conf, pred_cls, target_cls)
        expect = np.array([[1.]])
        self.assertTrue((f1 == expect).all())

    def test_f1_two_class(self):
        """Test F1-score for two class
        """
        correct = np.array([[False], [True], [False]])
        conf = np.array([0.654, 0.702, 0.432])
        pred_cls = np.array([0., 0., 1.])
        target_cls = np.array([1. , 0., 0.])
        _, _, _, f1, _ = ap_per_class(correct, conf, pred_cls, target_cls)
        expect = np.array([[0.5], [0.]])
        self.assertTrue((f1 == expect).all())

if __name__ == "__main__":
    unittest.main()
