from matplotlib.axes import Axes
import numpy as np
import math
from skimage.draw import line_aa
import matplotlib.pyplot as plt

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
        # upper and lower bound of drivable area orthogonal to
        # waypoint orientation.
        # Upper bound: free drivable area to the left of center-line in m
        # Lower bound: free drivable area to the right of center-line in m
        self.lb = None
        self.ub = None
        self.static_border_cells = None
        self.dynamic_border_cells = None
        
    def __str__(self):
        return f"Waypoint(x={self.x}, y={self.y}, psi={self.psi}, kappa={self.kappa})"

    def __sub__(self, other):
        """
        Overload subtract operator. Difference of two waypoints is equal to
        their euclidean distance.
        :param other: subtrahend
        :return: euclidean distance between two waypoints
        """
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5
    

class Track:
    def __init__(self, wp_x, wp_y, resolution, smoothing, width):
        """
        Track object. Create a reference trajectory from specified
        corner points with given resolution. Smoothing around corners can be
        applied. Waypoints represent center-line of the path with specified
        maximum width to both sides.
        :param map: map object on which path will be placed
        :param wp_x: x coordinates of corner points in global coordinates
        :param wp_y: y coordinates of corner points in global coordinates
        :param resolution: resolution of the path in m/wp
        :param smoothing: number of waypoints used for smoothing the
        path by averaging neighborhood of waypoints
        :param max_width: maximum width of path to both sides in m
        """
        
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
        # Length of path
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
            # first waypoint
            if wp_id == 0:
                kappa = 0
            else:
                prev_wp = np.array(waypoint_coordinates[wp_id - 1])
                dif_behind = current_wp - prev_wp
                angle_behind = np.arctan2(dif_behind[1], dif_behind[0])
                angle_dif = np.mod(psi - angle_behind + math.pi, 2 * math.pi) \
                            - math.pi
                kappa = angle_dif / (dist_ahead + self.eps)

            waypoints.append(Waypoint(x, y, psi, kappa))

        return waypoints
    
    def _compute_length(self):
        """
        Compute length of center-line path as sum of euclidean distance between
        waypoints.
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
        # colors = [wp.v_ref for wp in self.waypoints]
        axis.scatter(wp_x, wp_y, s=10)
    

# Specify waypoints
# wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25,
#         1.25, -0.75, -0.75, -0.25]
# wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0,
#         -1.5, -1.5]