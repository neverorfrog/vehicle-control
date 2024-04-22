from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import scipy
from scipy.integrate import trapezoid
import scipy.interpolate
from utils.common_utils import wrap
from typing import List
import casadi as ca
import matplotlib.patches as plt_patches

class Waypoint:
    def __init__(self, x, y, psi):
        """
        Waypoint object containing x, y location in global coordinate system,
        orientation of waypoint psi.  
        :param x: x position in global coordinate system | [m] 
        :param y: y position in global coordinate system | [m] 
        :param psi: orientation of waypoint | [rad]
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
    
class Obstacle:
    def __init__(self, cx, cy, s, ey, radius):
        """
        Constructor for a circular obstacle to be placed on a map.
        :param cx: x coordinate of center of obstacle in world coordinates
        :param cy: y coordinate of center of obstacle in world coordinates
        :param radius: radius of circular obstacle in m
        """
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.s = s
        self.ey = ey
        
    def __repr__(self) -> str:
        return f"Obstacle(cx={self.cx}, cy={self.cy}, radius={self.radius})"

    def plot(self, axis: Axes):
        """
        Display obstacle on given axis.
        """
        # Draw circle
        circle = plt_patches.Circle(xy=(self.cx, self.cy), radius=self.radius, color='#2E4053', zorder=20)
        axis.add_patch(circle)
    

class Track:
    def __init__(self, corners, smoothing, resolution, width, obstacle_data):
        """
        Track object containing a list of waypoints and a spline constructed
        """
        self.width = width
        self.resolution = resolution
        self.smoothing = smoothing
        self.waypoints: List[Waypoint] = self._construct_path(corners)
        self.n_waypoints = len(self.waypoints)
        self.x, self.y, self.dx_ds, self.dy_ds, self.ddx_ds, self.ddy_ds = self._construct_spline()
        self.curvatures = self._precompute_curvatures()
        self._divide_track()
        self.obstacles: List[Obstacle] = self._construct_obstacles(obstacle_data)
        
    def rel2glob(self, s, ey, epsi):
        orientation = self.get_orientation(s)
        x = self.x(s) - ca.sin(orientation) * ey
        y = self.y(s) + ca.cos(orientation) * ey
        psi = wrap(orientation + epsi)
        return x.full().squeeze(), y.full().squeeze(), psi.full().squeeze()
        
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
    
    def _construct_obstacles(self, obstacle_data):
        obstacles = []
        
        # obstacle list
        for obstacle_data in obstacle_data:
            s, ey, radius = obstacle_data
            x, y, _ = self.rel2glob(s, ey, 0)
            obstacles.append(Obstacle(x, y, s, ey, radius))
            
        # 2D binary occupancy grid
        s_values = np.arange(0, self.length - 0.1, 0.5)
        ey_values = np.arange(0, self.width, 0.01) - self.width/2
        S,EY = np.meshgrid(s_values,ey_values,indexing='ij')
        Z = np.zeros_like(S)
        orientation = self.get_orientation(S)
        X = (self.x(S) - ca.sin(orientation) * EY).full().squeeze()
        Y = (self.y(S) + ca.cos(orientation) * EY).full().squeeze()
        for obs in obstacles:
            Z += np.sqrt((X - obs.cx)**2 + (Y - obs.cy)**2) <= (obs.radius+1)**2
        data_flat = Z.ravel(order='F')
        self.occupancy: ca.Function  = ca.interpolant('occupancy', 'linear', [s_values,ey_values], data_flat, {'inline': True})
        return obstacles
            
    def _precompute_curvatures(self):
        self.ds = 0.05
        curvatures = []
        s_values = np.arange(0, self.length - 0.1, self.ds)
        for s in s_values:
            curvatures.append(self.get_curvature(s).full().squeeze().item())
        k_spline = ca.interpolant('curvatures', 'bspline', [s_values], curvatures, {'degree': [3]})
        s = ca.MX.sym('s')
        self.k = ca.Function("curvature",[s],[k_spline(s)])
        self.dk_ds = ca.Function("dk_ds",[s],[ca.jacobian(self.k(s),s)])
        
    def _divide_track(self):
        '''Divide the track into segments (straight and curve)'''
        segments = []
        s_values = np.arange(0, self.length-0.1, self.ds) 
        eps = 1e-7 #curve threshold
        start = 0
        is_curve = False #by default the starting segment is straight
        max_curv = 0
        
        for s in s_values:
            curv = self.k(s)
            
            # curve is starting while on a straight segment
            if abs(curv) > eps and not is_curve:
                if s - start > 1:
                    is_curve = True
                    segments.append([start, s, 0])
                    start = s
            
            # take track of the maximum curvature if we are in a curve
            if is_curve:
                if curv > max_curv:
                    max_curv = curv
            
            # curve ends
            if abs(curv) < eps and is_curve == True:
                if s - start > 1:
                    is_curve = False
                    segments.append([start, s, max_curv.full().squeeze().item()])
                    start = s
            
            #track ends
            if s >= self.length-0.15:
                segments.append([start, s, False])
                
        self.segments = segments
           
    def _construct_spline(self):
        # waypoint list
        waypoints_x = [waypoint.x for waypoint in self.waypoints]
        waypoints_y = [waypoint.y for waypoint in self.waypoints]
        s_values = np.arange(len(waypoints_x))  # Assuming waypoints are evenly spaced along the track
        
        # spline function definition
        x_spline = scipy.interpolate.InterpolatedUnivariateSpline(s_values, waypoints_x, k=3, ext=3)
        y_spline = scipy.interpolate.InterpolatedUnivariateSpline(s_values, waypoints_y, k=3, ext=3)
        
        x_values = []
        y_values = []
        for s in s_values:
            x_values.append(x_spline(s))
            y_values.append(y_spline(s))
        
        x_spline = ca.interpolant('x_spline', 'bspline', [s_values], x_values)
        y_spline = ca.interpolant('y_spline', 'bspline', [s_values], y_values)
        
        #computing the length
        s = ca.MX.sym('s')
        x = ca.Function("x_pos",[s],[x_spline(s)])
        y = ca.Function("y_pos",[s],[y_spline(s)])
        dx_ds = ca.Function("dx_ds",[s],[ca.jacobian(x(s),s)])
        dy_ds = ca.Function("dy_ds",[s],[ca.jacobian(y(s),s)])
        one_lap_range = np.arange(0, len(self.waypoints))
        compute_segment_length = lambda s: np.sqrt(dx_ds(s)**2 + dy_ds(s)**2).full().squeeze()
        self.length = trapezoid(compute_segment_length(one_lap_range), one_lap_range,dx=0.1)
        
        # redefining casadi functions (because s has to be spread from range [0,length] to [0,len(waypoints)])
        x = ca.Function("x_pos",[s],[x_spline((s/self.length) * len(self.waypoints))])
        y = ca.Function("y_pos",[s],[y_spline((s/self.length) * len(self.waypoints))])
        dx_ds = ca.Function("dx_ds",[s],[ca.jacobian(x(s),s)])
        dy_ds = ca.Function("dy_ds",[s],[ca.jacobian(y(s),s)])
        ddx_ds = ca.Function("ddx_ds",[s],[ca.jacobian(dx_ds(s),s)])
        ddy_ds = ca.Function("ddy_ds",[s],[ca.jacobian(dy_ds(s),s)])
        return x, y, dx_ds, dy_ds, ddx_ds, ddy_ds
        
    def _construct_path(self, corners):
        """
        Construct path from given waypoints.
        :return: list of waypoint objects
        """
        wp_x = []
        wp_y = []
        
        for i in range(len(corners)-1):
            start = [corners[i][0], corners[i][1]]
            end = [corners[i + 1][0], corners[i + 1][1]]
            distance = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            n_wp = int(distance/self.resolution)
            wp_x.extend(np.linspace(start[0], end[0], n_wp, endpoint=False).tolist())
            wp_y.extend(np.linspace(start[1], end[1], n_wp, endpoint=False).tolist())
            
        # Smooth path
        wp_xs = []
        wp_ys = []
        for wp_id in range(0, len(wp_x)):
            if wp_id < self.smoothing:
                wp_xs.append(wp_x[wp_id])
                wp_ys.append(wp_y[wp_id])
            elif wp_id > len(wp_x) - self.smoothing - 1:
                wp_xs.append(wp_x[wp_id])
                wp_ys.append(wp_y[wp_id])
            else:
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