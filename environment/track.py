# inspired by https://github.com/matssteinweg/Multi-Purpose-MPC

from matplotlib.axes import Axes
import numpy as np
from scipy.integrate import trapezoid
from utils.common_utils import wrap
from typing import List
import casadi as ca

class Waypoint:
    def __init__(self, x, y, psi):
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
        'x position of waypoint'
        self.y = y
        'y position of waypoint'
        self.psi = psi
        'path curvature at waypoint'
        self.v_ref = None
        'reference velocity at this waypoint according to speed profile'
        self.lb = None
        'left bound on drivable area wrt center line(track is anticlockwise)'
        self.rb = None
        'right bound on drivable area wrt center line(track is anticlockwise)'
        
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.psi
        
    def __str__(self):
        return f"Waypoint(x={self.x}, y={self.y}, psi={self.psi}, v_ref={self.v_ref})"

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
        :param smoothing: number of waypoints used for smoothing the path by averaging neighborhood of waypoints
        :param width: width of path to both sides in m
        """
        self.width = width
        self.resolution = resolution
        self.smoothing = smoothing
        self.waypoints: List[Waypoint] = self._construct_path(wp_x, wp_y)
        self.n_waypoints = len(self.waypoints)
        self._construct_spline()
        
    def get_curvature(self, s):
        '''Get curvature (inverse of curvature radius) of a point along the spline'''
        dx_ds = self.dx_ds(s)
        dy_ds = self.dy_ds(s)
        ddx_ds = self.ddx_ds(s)
        ddy_ds = self.ddy_ds(s)
        denom = ca.power(dx_ds**2 + dy_ds**2, 1.5)
        num = dx_ds * ddy_ds - ddx_ds * dy_ds
        curvature = ca.if_else(ca.fabs(denom) < ca.DM(1e-2), ca.DM(0.), num/denom)
        # return ca.if_else(ca.fabs(curvature) < ca.DM(1e-10), ca.DM(0.), curvature)
        return curvature
    
    def get_orientation(self, s):
        '''Get orientation wrt horizontal line of a point along the spline'''
        dx_ds = self.dx_ds(s)
        dy_ds = self.dy_ds(s)
        magnitude = np.sqrt(dx_ds**2 + dy_ds**2)
        tangent_x = dx_ds / magnitude
        tangent_y = dy_ds / magnitude
        return np.arctan2(tangent_y, tangent_x)
        
    def _construct_spline(self):
        # waypoint list
        waypoints_x = [waypoint.x for waypoint in self.waypoints]
        waypoints_y = [waypoint.y for waypoint in self.waypoints]
        s_values = np.arange(len(waypoints_x))  # Assuming waypoints are evenly spaced along the track
        
        # spline function definition
        self.x_spline = ca.interpolant('x_spline', 'bspline', [s_values], waypoints_x)
        self.y_spline = ca.interpolant('y_spline', 'bspline', [s_values], waypoints_y)
        
        #computing the length
        s = ca.MX.sym('s')
        x = ca.Function("x_pos",[s],[self.x_spline(s)])
        y = ca.Function("y_pos",[s],[self.y_spline(s)])
        dx_ds = ca.Function("dx_ds",[s],[ca.jacobian(x(s),s)])
        dy_ds = ca.Function("dy_ds",[s],[ca.jacobian(y(s),s)])
        one_lap_range = np.arange(0, len(self.waypoints))
        compute_segment_length = lambda s: np.sqrt(dx_ds(s)**2 + dy_ds(s)**2).full().squeeze()
        self.length = trapezoid(compute_segment_length(one_lap_range), one_lap_range,dx=0.0001)
        
        # redefining casadi functions (because s has to be normalized)
        self.x = ca.Function("x_pos",[s],[self.x_spline(s * len(self.waypoints) / self.length)])
        self.y = ca.Function("y_pos",[s],[self.y_spline(s * len(self.waypoints) / self.length)])
        self.dx_ds = ca.Function("dx_ds",[s],[ca.jacobian(self.x(s),s)])
        self.dy_ds = ca.Function("dy_ds",[s],[ca.jacobian(self.y(s),s)])
        self.ddx_ds = ca.Function("ddx_ds",[s],[ca.jacobian(self.dx_ds(s),s)])
        self.ddy_ds = ca.Function("ddy_ds",[s],[ca.jacobian(self.dy_ds(s),s)])
        
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
            
        # closing the circuit
        wp_xs.append(wp_xs[0])
        wp_ys.append(wp_ys[0])
        wp_xs.append(wp_xs[0])
        wp_ys.append(wp_ys[0])
        
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
        for wp_id in range(len(waypoint_coordinates)-1):

            # Get start and goal waypoints
            current_wp = np.array(waypoint_coordinates[wp_id])
            next_wp = np.array(waypoint_coordinates[wp_id + 1])

            # Difference vector
            dif_ahead = next_wp - current_wp

            # Angle ahead
            psi = np.arctan2(dif_ahead[1], dif_ahead[0])

            # Get x and y coordinates of current waypoint
            x, y = current_wp[0], current_wp[1]

            new_waypoint = Waypoint(x, y, psi)
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
    
    def plot(self, axis: Axes):
        """
        Display path object on current figure.
        :param display_drivable_area: If True, display arrows indicating width
        of drivable area
        """
        # Get x and y locations of border cells for left and right bound and closing the circuit
        lb_x = np.array([wp.lb[0] for wp in self.waypoints] + [self.waypoints[0].lb[0]])
        lb_y = np.array([wp.lb[1] for wp in self.waypoints] + [self.waypoints[0].lb[1]])
        rb_x = np.array([wp.rb[0] for wp in self.waypoints] + [self.waypoints[0].rb[0]])
        rb_y = np.array([wp.rb[1] for wp in self.waypoints] + [self.waypoints[0].rb[1]])
        
        # Plot road
        axis.plot(rb_x, rb_y, color='red')
        axis.plot(lb_x, lb_y, color='red')
        axis.fill(lb_x, lb_y, "grey",alpha=0.3)
        axis.fill(rb_x, rb_y, "w",alpha=0.9)