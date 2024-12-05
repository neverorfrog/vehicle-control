import casadi as ca
import numpy as np
from casadi import cos, tanh

from vehicle_control.models.racing_car import RacingCar
from vehicle_control.utils.fancy_vector import FancyVector
from vehicle_control.utils.integrators import Euler


class DynamicPointMass(RacingCar):

    @classmethod
    def create_state(cls, *args, **kwargs):
        return DynamicPointMassState(*args, **kwargs)

    @classmethod
    def create_action(cls, *args, **kwargs):
        return DynamicPointMassAction(*args, **kwargs)

    # TODO: if there is some value to print, do it here
    def print(self, state, input):
        V, s, ey, epsi, t = state
        Fx, Fy = input
        print("##########################")

    def _init_model(self):
        car = self.config["car"]
        env = self.config["env"]

        # =========== State and auxiliary variables ===================================
        V, s, ey, epsi, t = self.state.variables
        curvature = ca.SX.sym("curvature")
        ds = ca.SX.sym("ds")
        dt = ca.SX.sym("dt")
        g = 9.88

        # ====== Input Model ==========================================================
        Fx, Fy = self.input.variables
        Xd = car.Xd
        Xdf = Xd.f
        Xdr = Xd.r
        # drive distribution
        Xb = car.Xb
        Xbf = Xb.f
        Xbr = Xb.r
        # brake distribution

        Xf = (Xdf - Xbf) / 2 * tanh(2 * (Fx / 1000 + 0.5)) + (Xdf + Xbf) / 2
        self.Xf = ca.Function("Xf", [Fx], [Xf]).expand()
        Fx_f = Fx * Xf
        self.Fx_f = ca.Function("Fx_f", [Fx], [Fx_f]).expand()

        Xr = (Xbr - Xdr) / 2 * tanh(-2 * (Fx / 1000 + 0.5)) + (Xdr + Xbr) / 2
        self.Xr = ca.Function("Xr", [Fx], [Xr]).expand()
        Fx_r = Fx * Xr
        self.Fx_r = ca.Function("Fx_r", [Fx], [Fx_r]).expand()

        # ================= Normal Load ================================================
        a = car.a
        b = car.b
        l = car.l  # noqa: E741
        m = car.m
        h = car.h
        theta = env.theta
        phi = env.phi
        Av2 = env.Av2

        Fz_f = (b / l) * m * (g * cos(theta) * cos(phi) + Av2 * V**2) - h * Fx / l
        self.Fz_f = ca.Function("Fz_f", [V, Fx], [Fz_f]).expand()

        Fz_r = (a / l) * m * (g * cos(theta) * cos(phi) + Av2 * V**2) + h * Fx / l
        self.Fz_r = ca.Function("Fz_f", [V, Fx], [Fz_r]).expand()

        # ===================== Differential Equations ===================================
        Fb = 0  # -p.m*g*ca.cos(theta)*ca.sin(phi) TODO if you want to change the angle modify this
        Frr = (
            env.Frr
        )  # env['Crr']*Fn #rolling resistance = coefficient*normal force (not specified in the paper)
        Fd = Frr + env.Cd * (V**2)  # p.m*g*ca.sin(theta)

        # TEMPORAL transition (equations 1a to 1f)
        V_dot = (Fx - Fd) / m
        s_dot = (V * ca.cos(epsi)) / (1 - curvature * ey)
        ey_dot = V * ca.sin(epsi)
        epsi_dot = (Fy + Fb) / (m * V) - curvature * s_dot
        t_dot = 1
        state_dot = ca.vertcat(V_dot, s_dot, ey_dot, epsi_dot, t_dot)
        time_integrator = Euler(
            self.state.syms, self.input.syms, curvature, state_dot, dt
        )
        self._temporal_transition = time_integrator.step

        # SPATIAL transition (equations 41a to 41f)
        V_prime = V_dot / s_dot
        s_prime = 1
        ey_prime = ey_dot / s_dot
        epsi_prime = epsi_dot / s_dot
        t_prime = t_dot / s_dot
        state_prime = ca.vertcat(V_prime, s_prime, ey_prime, epsi_prime, t_prime)
        space_integrator = Euler(
            self.state.syms, self.input.syms, curvature, state_prime, ds
        )
        self._spatial_transition = space_integrator.step

    @property
    def transition(self):
        return self._temporal_transition

    @property
    def spatial_transition(self):
        return self._spatial_transition


class DynamicPointMassAction(FancyVector):
    def __init__(self, Fx=0.0, Fy=0.0):
        """
        :param Fx: longitudinal force | [m/s^2]
        :param Fy: lateral force | [m/s^2]
        """
        self._values = np.array([Fx, Fy])
        self._keys = ["Fx", "Fy"]
        self._syms = ca.vertcat(
            *[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))]
        )
        self._labels = [r"$F_x$", r"$F_y$"]

    @property
    def Fx(self):
        return self.values[0]

    @Fx.setter
    def Fx(self, value: float):
        assert isinstance(value, float)
        self.values[0] = value

    @property
    def Fy(self):
        return self.values[1]

    @Fy.setter
    def Fy(self, value: float):
        assert isinstance(value, float)
        self.values[1] = value


class DynamicPointMassState(FancyVector):
    def __init__(self, V=0.0, s=0.0, ey=0.0, epsi=0.0, t=0.0):
        """
        :param V: longitudinal velocity in global coordinate system | [m/s]
        :param s: curvilinear abscissa | [m]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._values = np.array([V, s, ey, epsi, t])
        self._keys = ["V", "s", "ey", "epsi", "t"]
        self._syms = ca.vertcat(
            *[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))]
        )
        self.delta = 0  # fictituous steering angle (always zero)

    @property
    def V(self):
        return self.values[0]

    @property
    def s(self):
        return self.values[1]

    @property
    def ey(self):
        return self.values[2]

    @ey.setter
    def ey(self, value):
        self.values[2] = value

    @property
    def epsi(self):
        return self.values[3]

    @epsi.setter
    def epsi(self, value):
        self.values[3] = value

    @property
    def t(self):
        return self.values[4]
