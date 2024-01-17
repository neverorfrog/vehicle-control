import numpy as np
import numpy.linalg as la


class Track():
    """map object
    Attributes:
        getGlobalPosition: convert position from (s, ey) to (X,Y)
    """
    def __init__(self):
        # L-shaped track
        self.halfWidth = 0.4
        self.slack = 0.45
        lengthCurve = 4.5
        spec = np.array([[1.0, 0],
                         [lengthCurve, lengthCurve / np.pi],
                         # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
                         [lengthCurve / 2, -lengthCurve / np.pi],
                         [lengthCurve, lengthCurve / np.pi],
                         [lengthCurve / np.pi * 2, 0],
                         [lengthCurve / 2, lengthCurve / np.pi]])