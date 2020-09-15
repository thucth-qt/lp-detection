import json
import os

from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from ailibs.detector.yolov3.LicensePlateDetector import LicensePlateDetector
from ailibs.detector.yolov3.VehicleDetector import VehicleDetector
from ailibs.utils.torch_utils import select_device
from apis.backend.api.utils import *

# Setup device
device = select_device(device='')

# Declare vehicle detector instance
vhc_detector = VehicleDetector(weights=os.path.join("ailibs_data", "weights", "yolov3.pt"),
                                cfg=os.path.join("ailibs", "detector", "yolov3", "cfg", "yolov3.cfg"),
                                names=os.path.join("ailibs", "detector", "yolov3", "cfg", "coco.names"),
                                device=device)

# Declare license plate detector instance
lp_detector = LicensePlateDetector(weights=os.path.join("ailibs_data", "weights", "lp.pt"),
                                    cfg=os.path.join("ailibs", "detector", "yolov3", "cfg", "yolov3-lp.cfg"),
                                    names=os.path.join("ailibs", "detector", "yolov3", "cfg", "lp.names"),
                                    device=device)


# Create your views here.
class VehicleDetection(APIView):
    """Vehicle detection API
    """

    def post(self, request):
        """POST request method for vehicle detection

        Parameters
        ----------
        request : rest_framework.request.Request
            request
        """
        
        if request.data["image"]:
            # Read image from body request
            image_base64 = request.data["image"]
            image = read_image_b64(image_base64)

            # Detect vehicle
            vhc_pos = vhc_detector.detect(image)

            # Convert list of OutputPrediction objects to JSON
            res_data = dict()
            for idx, vhc in enumerate(vhc_pos):
                res_data[str(idx)] = vhc.__dict__

            return Response(data=res_data, status=status.HTTP_200_OK)

        return Response(data={"message": "'image' field not found"}, status=status.HTTP_400_BAD_REQUEST)


class LicensePlateInVehicleDetection(APIView):
    """License plate detection in vehicle API
    """

    def post(self, request):
        """POST request method for license plate detection in vehicle

        Parameters
        ----------
        request : rest_framework.request.Request
            request
        """
        
        if request.data["image"]:
            # Read image from body request
            image_base64 = request.data["image"]
            image = read_image_b64(image_base64)

            # Detect license plate
            lp_pos = lp_detector.detect(image)

            # Convert list of OutputPrediction objects to JSON
            res_data = dict()
            for idx, lp in enumerate(lp_pos):
                res_data[str(idx)] = lp.__dict__

            return Response(data=res_data, status=status.HTTP_200_OK)

        return Response(data={"message": "'image' field not found"}, status=status.HTTP_400_BAD_REQUEST)


class LicensePlateDetection(APIView):
    """License plate detection API
    """

    def post(self, request):
        """POST request method for vehicle detection

        Parameters
        ----------
        request : rest_framework.request.Request
            request
        """
        
        if request.data["image"]:
            # Read image from body request
            image_base64 = request.data["image"]
            image = read_image_b64(image_base64)

            # Detect vehicle in the image
            vhc_pos = vhc_detector.detect(image)

            lps = []
            for idx, vhc_out in enumerate(vhc_pos):
                # Positions of vehicle
                vhc_tl = vhc_out.tl
                vhc_br = vhc_out.br

                # Get vehicle image from origin image
                vhc_img = image[vhc_tl[1]:vhc_br[1], vhc_tl[0]:vhc_br[0]]

                # Detect license plate in the vehicle image
                lp_pos = lp_detector.detect(vhc_img, plot=True)
                cv2.imwrite("vehicle_%s.png" % str(idx), vhc_img)

                for idx in range(len(lp_pos)):
                    lp_pos[idx].tl, lp_pos[idx].br = scale(lp_pos[idx].tl, lp_pos[idx].br, vhc_tl)

                lps.extend(lp_pos)

            # Convert list of OutputPrediction objects to JSON
            res_data = dict()
            for idx, lp in enumerate(lps):
                res_data[str(idx)] = lp.__dict__
    
            return Response(data=res_data, status=status.HTTP_200_OK)
                    
        return Response(data={"message": "'image' field not found"}, status=status.HTTP_400_BAD_REQUEST)
