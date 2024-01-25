# inspired by https://github.com/matssteinweg/Multi-Purpose-MPC

from typing import Union
import numpy as np
from abc import abstractmethod
from abc import ABC
import casadi as ca
from casadi import cos, sin, tan

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
    def values(self): pass
    
    @property
    @abstractmethod
    def keys(self): pass
    
    @property
    @abstractmethod
    def syms(self): pass
    
    @property
    def variables(self): return [self.syms[i] for i in range(len(self.values))]
    
    def __getitem__(self, key: Union[int, str]):
        if isinstance(key, int):
            return self.values[key]
        elif isinstance(key, str):
            index = self.keys.index(key)
            return self.syms[index]

    def __setitem__(self, key: Union[int, str], value):
        if isinstance(key, int):
            self.values[key] = value
        elif isinstance(key, str):
            self.syms[self.keys.index(key)] = value

    def __len__(self):
        return len(self.values)
    
    def __str__(self):
        return {key: f"{value:.2f}" for key,value in dict(zip(self.keys, self.values)).items()}.__str__()
    
    def __add__(self, other):
        """
        Overload + operator.
        :param other: numpy array to be added to state vector
        """
        assert isinstance(other, (self.__class__)), "You can only sum two same states"
        tobesummed = other.state if isinstance(other, State) else other
        new_state = self.state + tobesummed
        return self.__class__.create(*new_state)

class KinematicCarState(State):
    def __init__(self, x = 0.0, y = 0.0, v = 0.0, psi = 0.0, delta = 0.0, s = 0.0, ey = 0.0, epsi = 0.0, t = 0.0):
        """
        :param x: x position in global coordinate system | [m]
        :param y: y position in global coordinate system | [m]
        :param v: velocity in global coordinate system | [m/s]
        :param psi: yaw angle | [rad]
        :param delta: steering angle | [rad]
        :param s: curvilinear abscissa | [m]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._values = np.array([x,y,v,psi,delta,s,ey,epsi,t])
        self._keys = ['x', 'y', 'v','psi','delta','s','ey','epsi','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
     
    @property
    def x(self): return self.values[0] 
      
    @property
    def y(self): return self.values[1]
    
    @property
    def v(self): return self.values[2] 
      
    @property
    def psi(self): return self.values[3]
    
    @property
    def delta(self): return self.values[4]
      
    @property
    def s(self): return self.values[5]
    
    @property
    def ey(self): return self.values[6]
    
    @ey.setter
    def ey(self, value): self.values[6] = value
    
    @property
    def epsi(self): return self.values[7]
    
    @epsi.setter
    def epsi(self, value): self.values[7] = value
    
    @property
    def t(self): return self.values[8]
    
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)