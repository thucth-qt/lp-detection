import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def read_image_b64(base64_string):
    """Convert base64 image to numpy array

    Args:
        base64_string (bytes): image base 64

    Returns:
        numpy.ndarray: image
    """
    buf = BytesIO()
    buf.write(base64.b64decode(base64_string))
    pimg = Image.open(buf)

    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def scale(tl, br, vhc_tl):
    """Convert bbox's position of vehicle image to original image

    Args:
        tl (tuple): top left position of bbox
        br (tuple): bottom right position of bbox
        vhc_tl (tuple): top left position of vehicle object

    Returns:
        tuple: top left and bottom right after scaling
    """

    tl_h_new = tl[0] + vhc_tl[0]
    tl_w_new = tl[1] + vhc_tl[1]
    br_h_new = br[0] + vhc_tl[0]
    br_w_new = br[1] + vhc_tl[1]

    return (tl_h_new, tl_w_new), (br_h_new, br_w_new)
