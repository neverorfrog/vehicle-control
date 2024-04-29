from abc import ABC
import casadi as ca

class Integrator(ABC):
    def step(self, state, action, curvature, h):
        return self.discrete_ode(state, action, curvature, h)
    
    @property
    def discrete_ode(self):
        return self._discrete_ode #redefined by subclass

class Euler(Integrator):

    def __init__(self, state: ca.MX, action: ca.MX, curvature: ca.MX, f: ca.MX, h: ca.MX):
        f = ca.Function('f', [state, action, curvature], [f])
        k = f(state, action, curvature)
        x_next = state + h*k
        self._discrete_ode = ca.Function('f_discrete', [state, action, curvature, h], [x_next]).expand()


class RK4(Integrator):

    def __init__(self, state: ca.MX, action: ca.MX, curvature: ca.MX, f: ca.MX, h: ca.MX):
        f = ca.Function('f', [state, action, curvature], [f])
        k_1 = f(state, action, curvature)
        k_2 = f(state + 0.5 * h * k_1, action, curvature)
        k_3 = f(state + 0.5 * h * k_2, action, curvature)
        k_4 = f(state + h * k_3, action, curvature)
        state_next = state + h * (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        self._discrete_ode = ca.Function('f_discrete', [state, action, curvature, h], [state_next]).expand()

class RK2(Integrator):

    def __init__(self, state: ca.MX, action: ca.MX, curvature: ca.MX, f: ca.MX, h: ca.MX):
        f = ca.Function('f', [state, action, curvature], [f])
        k_1 = f(state, action, curvature)
        k_2 = f(state + 0.5 * h * k_1, action, curvature)
        state_next = state + h * k_2
        self._discrete_ode = ca.Function('f_discrete', [state, action, curvature, h], [state_next]).expand()


class CVODESIntegrator(): # TODO DOES NOT WORK

    def __init__(self, state: ca.MX, action: ca.MX, curvature: ca.MX, f: ca.MX, h: ca.MX):

        f = ca.Function('f', [state, action, curvature], [f])
        ode = {'x': state, 'p': action, 'ode': f}
        t0 = 0.
        tf = 0.4
        state_next = ca.integrator('F', 'cvodes', ode, t0, tf)
        self._discrete_ode = ca.Function('f_discrete', [state, action, curvature, h], [state_next])