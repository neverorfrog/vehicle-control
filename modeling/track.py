# adapted from https://github.com/urosolia/RacingLMPC/blob/master/src/fnc/simulator/Track.py
from matplotlib.axes import Axes
import numpy as np
from modeling.trajectory import Trajectory
from modeling.util import *

# A track is specified by a series of segments defined as the tuple [length,
# radius of curvature]. Given these segments we compute the (x, y) points of the
# track and the angle of the tangent vector (psi) at these points. For each
# segment we compute the (x, y, psi) coordinate at the last point of the
# segment. Furthermore, we compute also the cumulative s at the starting point
# of the segment at signed curvature In the end each segment will be defined by
# a tuple point_tangent = [x, y, psi, cumulative s, segment length, signed curvature]
ippodromo = lambda curve_length: np.array([
                [3.0, 0], 
                [curve_length, curve_length / np.pi],
                [3.0, 0],
                [curve_length, curve_length / np.pi]]) 

goggle = lambda curve_length: np.array([
                [1.0, 0],
                [curve_length, curve_length / np.pi],
                # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
                [curve_length / 2, -curve_length / np.pi],
                [curve_length, curve_length / np.pi],
                [curve_length / np.pi * 2, 0],
                [curve_length / 2, curve_length / np.pi]])

class Track(Trajectory):
    def __init__(self, freq=0.5):
        self.half_width = 0.4
        self.slack = 0.45
        curve_length = 4.5
        self.freq = freq
        self.spec = ippodromo(curve_length)       
        point_tangent = np.zeros((self.spec.shape[0] + 1, 6))
        for i in range(0, self.spec.shape[0]):
            if self.spec[i, 1] == 0.0: # current segment is a straight line
                point_tangent[i, :] = self.straight_line(i, point_tangent)
            else:   
                point_tangent[i, :] = self.curved_line(i, point_tangent)
          
        # last segment  
        xs = point_tangent[-2, 0]
        ys = point_tangent[-2, 1]
        xf = 0
        yf = 0
        psif = 0
        l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)
        new_line = np.array([xf, yf, psif, point_tangent[-2, 3] + point_tangent[-2, 4], l, 0])
        point_tangent[-1, :] = new_line
        
        # wrapping up
        self.track_length = point_tangent[-1, 3] + point_tangent[-1, 4] 
        self.point_tangent = point_tangent
    
    def update(self,t): # TODO
        p = self.get_global_position(t * self.freq * self.track_length, 0)
        pd = 0
        pdd = 0
        return {'p': p, 'pd':pd , 'pdd':pdd}    
        
    def straight_line(self,i, point_tangent):
        l = self.spec[i, 0] # Length of the segments
        
        ang = 0 if i == 0 else point_tangent[i - 1, 2] # Angle of the tangent vector at the starting point of the segment
        x = (0 if i == 0 else point_tangent[i-1, 0]) + l * np.cos(ang) # x coordinate of the last point of the segment
        y = (0 if i == 0 else point_tangent[i-1, 1]) + l * np.sin(ang) # y coordinate of the last point of the segment
        psi = ang  # Angle of the tangent vector at the last point of the segment
        
        if i == 0:
            new_line = np.array([x, y, psi, point_tangent[i, 3], l, 0])
        else:
            new_line = np.array([x, y, psi, point_tangent[i-1, 3] + point_tangent[i-1, 4], l, 0])
        
        return new_line
    
    def curved_line(self,i, point_tangent):
        l = self.spec[i, 0]                 # Length of the segment
        r = self.spec[i, 1]                 # Radius of curvature
        direction = 1 if r >= 0 else -1  
        ang = 0 if i == 0 else point_tangent[i - 1, 2] # Angle of the tangent vector at the starting point of the segment
        
        center_x = (0 if i == 0 else point_tangent[i-1, 0]) + \
            np.abs(r) * np.cos(ang + direction * np.pi / 2) # x coordinate center of circle
        center_y = (0 if i == 0 else point_tangent[i-1, 1]) + \
            np.abs(r) * np.sin(ang + direction * np.pi / 2) # y coordinate center of circle
        

        spanAng = l / np.abs(r)  # Angle spanned by the circle
        psi = wrap(ang + spanAng * np.sign(r))  # Angle of the tangent vector at the last point of the segment

        angleNormal = wrap((direction * np.pi / 2 + ang))
        angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
        x = center_x + np.abs(r) * np.cos(angle + direction * spanAng)  # x coordinate of the last point of the segment
        y = center_y + np.abs(r) * np.sin(angle + direction * spanAng)  # y coordinate of the last point of the segment

        if i == 0:
            new_line = np.array([x, y, psi, point_tangent[i, 3], l, 1 / r])
        else:
            new_line = np.array([x, y, psi, point_tangent[i-1, 3] + point_tangent[i-1, 4], l, 1 / r])
        
        return new_line
    
    def get_index(self, s):
        '''Compute the segment in which system is evolving'''
        conditions = np.array([ [s >= self.point_tangent[:, 3]] , [s < self.point_tangent[:, 3] + self.point_tangent[:, 4]] ]).squeeze()
        index = np.where(np.all(conditions, axis = 0))[0]
        i = 0 if len(index) < 1 else int(index.item()) # TODO better solution?
        return i
    
    def get_global_position(self, s, ey):
        """coordinate transformation from curvilinear reference frame (s, ey) to inertial reference frame (X, Y)
        (s, ey): position in the curvilinear reference frame
        """
        point_tangent = self.point_tangent
        
        # wrap s along the track
        while (s > self.track_length):
            s = s - self.track_length
        i = self.get_index(s)
        
        if point_tangent[i, 5] == 0.0:  # If segment is a straight line
            # Extract the first final and initial point of the segment
            xf = point_tangent[i, 0]
            yf = point_tangent[i, 1]
            xs = point_tangent[i - 1, 0]
            ys = point_tangent[i - 1, 1]
            psi = point_tangent[i, 2]

            # Compute the segment length
            deltaL = point_tangent[i, 4]
            reltaL = s - point_tangent[i, 3]

            # Do the linear combination
            x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
            y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
        else:
            r = 1 / point_tangent[i, 5]  # Extract curvature
            ang = point_tangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)
            
            # Compute the center of the arc
            direction = sign(r)
            center_x = point_tangent[i - 1, 0] + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
            center_y = point_tangent[i - 1, 1] + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle
            
            spanAng = (s - point_tangent[i, 3]) / (np.pi * np.abs(r)) * np.pi
            angleNormal = wrap((direction * np.pi / 2 + ang))
            angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
            x = center_x + (np.abs(r) - direction * ey) * np.cos(angle + direction * spanAng)  # x coordinate of the last point of the segment
            y = center_y + (np.abs(r) - direction * ey) * np.sin(angle + direction * spanAng)  # y coordinate of the last point of the segment
        return np.array([x,y])
    
    def get_curvature(self, s):
        """curvature computation
        s: curvilinear abscissa at which the curvature has to be evaluated
        """
        while (s > self.track_length):
            s = s - self.track_length
        i = self.get_index(s)
        curvature = self.point_tangent[i, 5]
        return curvature
    
    def plot(self, axis: Axes):
        points = int(np.floor(10 * (self.point_tangent[-1, 3] + self.point_tangent[-1, 4])))
        right_limit_points = np.zeros((points, 2))
        left_limit_points = np.zeros((points, 2))
        center_points = np.zeros((points, 2))
        for i in range(0, int(points)):
            left_limit_points[i, :] = self.get_global_position(i * 0.1, self.half_width)
            right_limit_points[i, :] = self.get_global_position(i * 0.1, -self.half_width)
            center_points[i, :] = self.get_global_position(i * 0.1, 0)
            
        axis.plot(self.point_tangent[:, 0], self.point_tangent[:, 1], 'o')
        axis.plot(center_points[:, 0], center_points[:, 1], '--')
        axis.plot(left_limit_points[:, 0], left_limit_points[:, 1], '-b')
        axis.plot(right_limit_points[:, 0], right_limit_points[:, 1], '-b')
   
