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


class Detector:
    """Abstract class for detector
    """
    def detect(self, image, plot=False):
        """Detection

        Parameters
        ----------
        image : numpy.ndarray or tensor
            input image
        plot: bool
            plot bounding box

        Raises
        ------
        NotImplementedError
            raise error if detect method hasn't implemented.
        """
        raise NotImplementedError("detect method hasn't implemented.")