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
    def __init__(self, corners, curves, smoothing, resolution, width = 0.4):
        """
        Track object containing a list of waypoints and a spline constructed
        """
        self.width = width
        self.resolution = resolution
        self.smoothing = smoothing
        self.waypoints: List[Waypoint] = self._construct_path(corners, curves)
        self.n_waypoints = len(self.waypoints)
        self._construct_spline()
        self.ds = 0.03
        self.curvatures = self._precompute_curvatures()
        
    def get_curvature(self, s):
        '''Get curvature (inverse of curvature radius) of a point along the spline'''
        s = ca.fmod(s,self.length) # need to module s (for successive laps)
        dx_ds = self.dx_ds(s)
        dy_ds = self.dy_ds(s)
        ddx_ds = self.ddx_ds(s)
        ddy_ds = self.ddy_ds(s)
        denom = ca.power(dx_ds**2 + dy_ds**2, 1.5)
        num = ca.fabs(dx_ds * ddy_ds - ddx_ds * dy_ds)
        curvature = num/denom
        return curvature
    
    def get_orientation(self, s):
        '''Get orientation wrt horizontal line of a point along the spline'''
        s = ca.fmod(s,self.length) # need to module s (for successive laps)
        dx_ds = self.dx_ds(s)
        dy_ds = self.dy_ds(s)
        magnitude = (dx_ds**2 + dy_ds**2)**0.5
        tangent_x = dx_ds / magnitude
        tangent_y = dy_ds / magnitude
        return np.arctan2(tangent_y, tangent_x)
    
    def get_speed(self, s): # TODO
        '''Get desired speed of a point along the spline'''
        s = ca.fmod(s,self.length) # need to module s (for successive laps)
        return 10 * (1 - self.get_curvature(s))
    
    def _precompute_curvatures(self):
        curvatures = []
        s_values = np.arange(0, self.length-0.1, self.ds)
        for s in s_values:
            curvatures.append(self.get_curvature(s).full().squeeze().item())
        curvatures = ca.interpolant('curvatures', 'bspline', [s_values], curvatures)
        return curvatures
        
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
        self.length = trapezoid(compute_segment_length(one_lap_range), one_lap_range,dx=0.1)
        
        # redefining casadi functions (because s has to be spread from range [0,length] to [0,len(waypoints)])
        self.x = ca.Function("x_pos",[s],[self.x_spline((s/self.length) * len(self.waypoints))])
        self.y = ca.Function("y_pos",[s],[self.y_spline((s/self.length) * len(self.waypoints))])
        self.dx_ds = ca.Function("dx_ds",[s],[ca.jacobian(self.x(s),s)])
        self.dy_ds = ca.Function("dy_ds",[s],[ca.jacobian(self.y(s),s)])
        self.ddx_ds = ca.Function("ddx_ds",[s],[ca.jacobian(self.dx_ds(s),s)])
        self.ddy_ds = ca.Function("ddy_ds",[s],[ca.jacobian(self.dy_ds(s),s)])
        
    def _construct_path(self, corners, curves):
        """
        Construct path from given waypoints.
        :return: list of waypoint objects
        """
        
        wp_x = []
        wp_y = []
        
        angle = 0 # TODO hardcodato
        for i in range(len(corners)-1):
            start = [corners[i][0], corners[i][1]]
            end = [corners[i + 1][0], corners[i + 1][1]]
            distance = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            # if curves[i] == False:
            n_wp = int(distance/self.resolution)
            wp_x.extend(np.linspace(start[0], end[0], n_wp, endpoint=False).tolist())
            wp_y.extend(np.linspace(start[1], end[1], n_wp, endpoint=False).tolist())
            # else:
            #     midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            #     radius = np.sqrt(distance**2 / 2)
            #     centerx = start[0] + radius * np.sin(angle)
            #     centery = start[1] + radius * np.cos(angle)
            #     start_angle = np.arctan2(centery - start[1], centerx - start[0])
            #     end_angle = np.arctan2(centery - end[1], centerx - end[0])
            #     x_list = []
            #     y_list = []
            #     n_wp = 250
            #     for i in range(n_wp):
            #         angle = start_angle + i * wrap((end_angle - start_angle)) / n_wp
            #         x = centerx - radius * np.cos(angle)
            #         y = centery - radius * np.sin(angle)
            #         x_list.append(x)
            #         y_list.append(y)
            #     wp_x.extend(x_list)
            #     wp_y.extend(y_list)
            # angle = np.arctan2(start[1] - end[1], end[0] - start[0])

        # Smooth path
        wp_xs = []
        wp_ys = []
        for wp_id in range(self.smoothing, len(wp_x) - self.smoothing):
            wp_xs.append(np.mean(wp_x[wp_id - self.smoothing : wp_id + self.smoothing + 1]))
            wp_ys.append(np.mean(wp_y[wp_id - self.smoothing : wp_id + self.smoothing + 1]))
            
        # closing the circuit
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
        wpx = np.array([wp.x for wp in self.waypoints])
        wpy = np.array([wp.y for wp in self.waypoints])
        axis.plot(wpx, wpy, 'k--')
        axis.plot(rb_x, rb_y, color='k')
        axis.plot(lb_x, lb_y, color='k')
        axis.fill(lb_x, lb_y, "grey",alpha=0.3)
        axis.fill(rb_x, rb_y, "w",alpha=0.9)