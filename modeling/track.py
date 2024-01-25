# inspired by https://github.com/matssteinweg/Multi-Purpose-MPC

from matplotlib.axes import Axes
import numpy as np
import math
from modeling.util import wrap

# Colors
DRIVABLE_AREA = '#BDC3C7'
WAYPOINTS = '#D0D3D4'
PATH_CONSTRAINTS = '#F5B041'

class Waypoint:
    def __init__(self, x, y, psi, kappa):
        """
        Waypoint object containing x, y location in global coordinate system,
        orientation of waypoint psi and local curvature kappa. Waypoint further
        contains an associated reference velocity computed by the speed profile
        and a path width specified by upper and lower bounds.
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param psi: orientation of waypoint | [rad]
        :param kappa: local curvature | [1 / m]
        """
        self.x = x
        self.y = y
        self.psi = psi
        self.kappa = kappa

        # Reference velocity at this waypoint according to speed profile
        self.v_ref = None

        # Information about drivable area at waypoint
        # left and right bound of drivable area orthogonal to
        # waypoint orientation. Track is anticlockwise.
        # Left bound: free drivable area to the left of center-line in m
        # Right bound: free drivable area to the right of center-line in m
        self.lb = None
        self.rb = None
        
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.psi
        
    def __str__(self):
        return f"Waypoint(x={self.x}, y={self.y}, psi={self.psi}, kappa={self.kappa}, v_ref={self.v_ref})"

    def __sub__(self, other):
        """
        Overload subtract operator. Difference of two waypoints is equal to
        their euclidean distance.
        :param other: subtrahend
        :return: euclidean distance between two waypoints
        """
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
    

class Track:
    def __init__(self, wp_x, wp_y, resolution, smoothing, width = 0.4):
        """
        Track object. Create a reference trajectory from specified
        corner points with given resolution. Smoothing around corners can be
        applied. Waypoints represent center-line of the path with specified
        maximum width to both sides. By default the track is anticlockwise.
        :param map: map object on which path will be placed
        :param wp_x: x coordinates of corner points in global coordinates
        :param wp_y: y coordinates of corner points in global coordinates
        :param resolution: resolution of the path in m/wp
        :param smoothing: number of waypoints used for smoothing the
        path by averaging neighborhood of waypoints
        :param width: width of path to both sides in m
        """
        self.width = width
        # Precision
        self.eps = 1e-12
        # Resolution of the path
        self.resolution = resolution
        # Look ahead distance for path averaging
        self.smoothing = smoothing
        # List of waypoint objects
        self.waypoints = self._construct_path(wp_x, wp_y)
        # Number of waypoints
        self.n_waypoints = len(self.waypoints)
        # Length and width of path
        self.length, self.segment_lengths = self._compute_length()
        
        
    def _construct_path(self, wp_x, wp_y):
        """
        Construct path from given waypoints.
        :param wp_x: x coordinates of waypoints in global coordinates
        :param wp_y: y coordinates of waypoints in global coordinates
        :return: list of waypoint objects
        """

        # Number of intermediate waypoints between one waypoint and the other
        distance = lambda i: np.sqrt((wp_x[i + 1] - wp_x[i]) ** 2 + (wp_y[i + 1] - wp_y[i]) ** 2)
        n_wp = [int(distance(i)/self.resolution) for i in range(len(wp_x) - 1)]
        
        # Construct waypoints with specified resolution
        gp_x, gp_y = wp_x[-1], wp_y[-1]
        x_list = lambda i: np.linspace(wp_x[i], wp_x[i+1], n_wp[i], endpoint=False).tolist()
        wp_x = [x_list(i) for i in range(len(wp_x)-1)]
        wp_x = [wp for segment in wp_x for wp in segment] + [gp_x]
        y_list = lambda i: np.linspace(wp_y[i], wp_y[i + 1], n_wp[i], endpoint=False).tolist()
        wp_y = [y_list(i) for i in range(len(wp_y) - 1)]
        wp_y = [wp for segment in wp_y for wp in segment] + [gp_y]

        # Smooth path
        wp_xs = []
        wp_ys = []
        for wp_id in range(self.smoothing, len(wp_x) - self.smoothing):
            wp_xs.append(np.mean(wp_x[wp_id - self.smoothing : wp_id + self.smoothing + 1]))
            wp_ys.append(np.mean(wp_y[wp_id - self.smoothing : wp_id + self.smoothing + 1]))

        # Construct list of waypoint objects
        waypoints = list(zip(wp_xs, wp_ys))
        
        waypoints = self._construct_waypoints(waypoints)

        return waypoints
    
    def _construct_waypoints(self, waypoint_coordinates):
        """
        Reformulate conventional waypoints (x, y) coordinates into waypoint
        objects containing (x, y, psi, kappa, ub, lb)
        :param waypoint_coordinates: list of (x, y) coordinates of waypoints in
        global coordinates
        :return: list of waypoint objects for entire reference path
        """

        # List containing waypoint objects
        waypoints = []

        # Iterate over all waypoints
        for wp_id in range(len(waypoint_coordinates) - 1):

            # Get start and goal waypoints
            current_wp = np.array(waypoint_coordinates[wp_id])
            next_wp = np.array(waypoint_coordinates[wp_id + 1])

            # Difference vector
            dif_ahead = next_wp - current_wp

            # Angle ahead
            psi = np.arctan2(dif_ahead[1], dif_ahead[0])
            
            # Distance to next waypoint
            dist_ahead = np.linalg.norm(dif_ahead, 2)

            # Get x and y coordinates of current waypoint
            x, y = current_wp[0], current_wp[1]

            # Compute local curvature at waypoint
            if wp_id == 0: # first waypoint
                kappa = 0
            else:
                prev_wp = np.array(waypoint_coordinates[wp_id - 1])
                dif_behind = current_wp - prev_wp
                angle_behind = np.arctan2(dif_behind[1], dif_behind[0])
                angle_dif = np.mod(psi - angle_behind + math.pi, 2 * math.pi) - math.pi
                kappa = angle_dif / (dist_ahead + self.eps)

            new_waypoint = Waypoint(x, y, psi, kappa)
            new_waypoint.v_ref = 3 - kappa # reference velocity
            self._set_bounds(new_waypoint) # set left and right bounds
            waypoints.append(new_waypoint)

        return waypoints
    
    def _set_bounds(self, waypoint: Waypoint):
        x,y,psi = waypoint
        orth_angle = wrap(psi+np.pi/2) #orthogonal angle to psi
        lbx = x - np.cos(orth_angle) * self.width/2
        rbx = x + np.cos(orth_angle) * self.width/2
        lby = y - np.sin(orth_angle) * self.width/2
        rby = y + np.sin(orth_angle) * self.width/2
        waypoint.lb = np.array([lbx,lby])
        waypoint.rb = np.array([rbx,rby])
        
    def _compute_length(self):
        """
        Compute length of center-line path as sum of euclidean distance between waypoints.
        :return: length of center-line path in m
        """
        distance = lambda wp_id: self.waypoints[wp_id+1] - self.waypoints[wp_id]
        segment_lengths = [0.0] + [distance(wp_id) for wp_id in range(len(self.waypoints)-1)]
        s = sum(segment_lengths)
        return s, segment_lengths
    
    def get_waypoint(self, wp_id):
        """
        Get waypoint corresponding to wp_id
        :param wp_id: unique waypoint ID
        :return: waypoint object
        """
        if wp_id >= self.n_waypoints:
            wp_id = np.mod(wp_id, self.n_waypoints)

        return self.waypoints[wp_id]
    
    def plot(self, axis: Axes, display_drivable_area=True):
        """
        Display path object on current figure.
        :param display_drivable_area: If True, display arrows indicating width
        of drivable area
        """

        # Get x and y coordinates for all waypoints
        wp_x = np.array([wp.x for wp in self.waypoints])
        wp_y = np.array([wp.y for wp in self.waypoints])
        
        # Plot waypoints
        axis.scatter(wp_x, wp_y, s=3)
        
        # Get x and y locations of border cells for left and right bound
        lb_x = np.array([wp.lb[0] for wp in self.waypoints])
        lb_y = np.array([wp.lb[1] for wp in self.waypoints])
        rb_x = np.array([wp.rb[0] for wp in self.waypoints])
        rb_y = np.array([wp.rb[1] for wp in self.waypoints])
        
        if display_drivable_area:
            axis.quiver(wp_x, wp_y, lb_x - wp_x, lb_y - wp_y, scale=1,
                    units='xy', width=0.2*self.resolution, color=DRIVABLE_AREA,
                    headwidth=1, headlength=0)
            axis.quiver(wp_x, wp_y, rb_x - wp_x, rb_y - wp_y, scale=1,
                    units='xy', width=0.2*self.resolution, color=DRIVABLE_AREA,
                    headwidth=1, headlength=0)
        
        # Closing the circuit
        lb_x = np.array([wp.lb[0] for wp in self.waypoints] + [self.waypoints[0].lb[0]])
        lb_y = np.array([wp.lb[1] for wp in self.waypoints] + [self.waypoints[0].lb[1]])
        rb_x = np.array([wp.rb[0] for wp in self.waypoints] + [self.waypoints[0].rb[0]])
        rb_y = np.array([wp.rb[1] for wp in self.waypoints] + [self.waypoints[0].rb[1]])
        
        axis.plot(rb_x, rb_y, color='red')
        axis.plot(lb_x, lb_y, color='red')