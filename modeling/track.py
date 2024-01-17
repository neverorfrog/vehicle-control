# taken from https://github.com/urosolia/RacingLMPC/blob/master/src/fnc/simulator/Track.py

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

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
    # Now given the above segments we compute the (x, y) points of the track and
    # the angle of the tangent vector (psi) at these points. For each segment we
    # compute the (x, y, psi) coordinate at the last point of the segment.
    # Furthermore, we compute also the cumulative s at the starting point of the
    # segment at signed curvature 
    # PointAndTangent = [x, y, psi, cumulative s, segment length, signed curvature]
        PointAndTangent = np.zeros((spec.shape[0] + 1, 6))
        
        # TODO continue implementation from github at top of code

    
if __name__ == "__main__":
    track = Track()