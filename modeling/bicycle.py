# inspired by https://github.com/matssteinweg/Multi-Purpose-MPC

import numpy as np


class Bicycle():
    def __init__(self, track, length, width, dt):
        """
        Bicycle Model.
        :param track: reference path object to follow
        :param length: length of car in m
        :param width: width of car in m
        :param dt: sampling time of model
        """
        
        # Precision
        self.eps = 1e-12

        # Car Parameters
        self.length = length
        self.width = width
        # self.safety_margin = self._compute_safety_margin()

        # Reference Path
        self.reference_path = track

        # Set initial distance traveled
        self.s = 0.0

        # Set sampling time
        self.dt = dt

        # Set initial waypoint ID
        self.wp_id = 0
        
    def drive(self, u):
        """
        Drive.
        :param u: input vector containing [v, delta]
        """

        # Get input signals
        v, delta = u

        # Compute temporal state derivatives
        x_dot = v * np.cos(self.temporal_state.psi)
        y_dot = v * np.sin(self.temporal_state.psi)
        psi_dot = v / self.length * np.tan(delta)
        temporal_derivatives = np.array([x_dot, y_dot, psi_dot])

        # Update spatial state (Forward Euler Approximation)
        self.temporal_state += temporal_derivatives * self.dt

        # Compute velocity along path
        s_dot = 1 / (1 - self.spatial_state.e_y * self.current_waypoint.kappa) \
                * v * np.cos(self.spatial_state.e_psi)

        # Update distance travelled along reference path
        self.s += s_dot * self.Ts