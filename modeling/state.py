# inspired by https://github.com/matssteinweg/Multi-Purpose-MPC

from typing import Union
import numpy as np
from abc import abstractmethod
from abc import ABC
import casadi as ca

class State(ABC):
    """
    Spatial State Vector - Abstract Base Class.
    """
    
    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs):
        pass
    
    @property
    @abstractmethod
    def state(self): pass
    
    @property
    @abstractmethod
    def members(self): pass
    
    @property
    @abstractmethod
    def state_sym(self): pass
    
    def __getitem__(self, key: Union[int, str]):
        if isinstance(key, int):
            return self.state[key]
        elif isinstance(key, str):
            index = self.members.index(key)
            return self.state_sym[index]

    def __setitem__(self, key: Union[int, str], value):
        if isinstance(key, int):
            self.state[key] = value
        elif isinstance(key, str):
            self.state_sym[self.members.index(key)] = value

    def __len__(self):
        return len(self.members)
    
    def __add__(self, other):
        """
        Overload + operator.
        :param other: numpy array to be added to state vector
        """
        assert isinstance(other, (self.__class__)), "You can only sum two same states"
        tobesummed = other.state if isinstance(other, State) else other
        new_state = self.state + tobesummed
        return self.__class__.create(*new_state)

class TemporalState(State):
    def __init__(self, x = 0.0, y = 0.0, psi = 0.0, delta = 0.0, s = 0.0):
        """
        Temporal State Vector containing car pose (x, y, psi)
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param psi: yaw angle | [rad]
        :param delta: steering angle | [rad]
        """
        self._state = np.array([x,y,psi,delta,s])
        self._state_sym = ca.vertcat(ca.SX.sym('x'),ca.SX.sym('y'),ca.SX.sym('psi'), ca.SX.sym('delta'), ca.SX.sym('s'))
        self._members = ['x', 'y', 'psi', 'delta','s']
    
    @property
    def state(self): return self._state
    
    @property
    def state_sym(self): return self._state_sym
    
    @property
    def members(self): return self._members
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __str__(self):
        return f'x: {self.state[0]}, y: {self.state[1]}, psi: {self.state[2]}, psi: {self.state[3]}, s: {self.state[4]}'
    
class SpatialState(State):
    def __init__(self, ey=0.0, epsi=0.0, t=0.0):
        """
        Simplified Spatial State Vector containing orthogonal deviation from
        reference path (e_y), difference in orientation (e_psi) and velocity
        :param e_y: orthogonal deviation from center-line | [m]
        :param e_psi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._state = np.array([ey,epsi,t])
        self._state_sym = ca.vertcat(ca.SX.sym('ey'),ca.SX.sym('epsi'), ca.SX.sym('t'))
        self._members = ['ey', 'epsi','t']
    
    @property
    def state(self): return self._state
    
    @property
    def state_sym(self): return self._state_sym
    
    @property
    def members(self): return self._members
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __str__(self):
        return f'ey: {self.state[0]}, epsi: {self.state[1]}, t: {self.state[2]}'