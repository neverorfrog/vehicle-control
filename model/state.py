from typing import Union
import numpy as np
from abc import abstractmethod
from abc import ABC
import casadi as ca

class FancyVector(ABC):
    """
    Abstract Base Class for States and Inputs
    In general it holds values and casadi variables (that's why it's fancy)
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
    
    def index(self, key): return self.keys.index(key)
    
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
        tobesummed = other.state if isinstance(other, FancyVector) else other
        new_state = self.state + tobesummed
        return self.__class__.create(*new_state)
    
class KinematicCarInput(FancyVector):
    def __init__(self, a = 0.0, w = 0.0):
        """
        :param a: longitudinal acceleration | [m/s^2]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([a,w])
        self._keys = ['a','w']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        
    @property
    def a(self): return self.values[0] 
      
    @property
    def w(self): return self.values[1]
    
    @a.setter
    def a(self,value: float): 
        assert isinstance(value, float)
        self.values[0] = value
    
    @w.setter
    def w(self,value: float): 
        assert isinstance(value, float)
        self.values[1] = value
        
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)

class KinematicCarState(FancyVector):
    def __init__(self, v = 0.0, delta = 0.0, s = 0.0, ey = 0.0, epsi = 0.0, t = 0.0):
        """
        :param v: velocity in global coordinate system | [m/s]
        :param psi: yaw angle | [rad]
        :param delta: steering angle | [rad]
        :param s: curvilinear abscissa | [m]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._values = np.array([v,delta,s,ey,epsi,t])
        self._keys = ['v','delta','s','ey','epsi','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
    
    @property
    def v(self): return self.values[0] 
      
    @property
    def delta(self): return self.values[1]
      
    @property
    def s(self): return self.values[2]
    
    @property
    def ey(self): return self.values[3]
    
    @ey.setter
    def ey(self, value): self.values[3] = value
    
    @property
    def epsi(self): return self.values[4]
    
    @epsi.setter
    def epsi(self, value): self.values[4] = value
    
    @property
    def t(self): return self.values[5]
    
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
class DynamicCarInput(FancyVector):
    def __init__(self, Fx = 0.0, w = 0.0):
        """
        :param Fx: longitudinal force | [m/s^2]
        :param w: steering angle rate | [rad/s]
        """
        self._values = np.array([Fx,w])
        self._keys = ['Fx','w']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        
    @property
    def Fx(self): return self.values[0] 
      
    @property
    def w(self): return self.values[1]
    
    @Fx.setter
    def Fx(self,value: float): 
        assert isinstance(value, float)
        self.values[0] = value
    
    @w.setter
    def w(self,value: float): 
        assert isinstance(value, float)
        self.values[1] = value
        
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
class DynamicCarState(FancyVector):
    def __init__(self, Ux = 0.0, Uy = 0.0, r = 0.0, delta = 0.0, s = 0.0, ey = 0.0, epsi = 0.0, t = 0.0):
        """
        :param Ux: longitudinal velocity in global coordinate system | [m/s]
        :param Uy: lateral velocity in global coordinate system | [m/s]
        :param r: yaw rate | [rad/s]
        :param delta: steering angle | [rad]
        :param s: curvilinear abscissa | [m]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._values = np.array([Ux,Uy,r,delta,s,ey,epsi,t])
        self._keys = ['Ux','Uy','r','delta','s','ey','epsi','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
    
    @property
    def Ux(self): return self.values[0] 
    
    @property
    def Uy(self): return self.values[1] 
    
    @property
    def r(self): return self.values[2]
    
    @property
    def delta(self): return self.values[3]
      
    @property
    def s(self): return self.values[4]
    
    @property
    def ey(self): return self.values[5]
    
    @ey.setter
    def ey(self, value): self.values[5] = value
    
    @property
    def epsi(self): return self.values[6]
    
    @epsi.setter
    def epsi(self, value): self.values[6] = value
    
    @property
    def t(self): return self.values[7]
    
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    


class DynamicPointMassInput(FancyVector):
    def __init__(self, Fx = 0.0, Fy = 0.0):
        """
        :param Fx: longitudinal force | [m/s^2]
        :param Fy: lateral force | [m/s^2]
        """
        self._values = np.array([Fx, Fy])
        self._keys = ['Fx', 'Fy']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
        
    @property
    def Fx(self): return self.values[0] 
    
    @Fx.setter
    def Fx(self,value: float): 
        assert isinstance(value, float)
        self.values[0] = value
        
    @property
    def Fy(self): return self.values[1] 
    
    @Fx.setter
    def Fy(self,value: float): 
        assert isinstance(value, float)
        self.values[1] = value
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
class DynamicPointMassState(FancyVector):
    def __init__(self, V = 0.0, s = 0.0, ey = 0.0, epsi = 0.0, t = 0.0):
        """
        :param V: longitudinal velocity in global coordinate system | [m/s]
        :param s: curvilinear abscissa | [m]
        :param ey: orthogonal deviation from center-line | [m]
        :param epsi: yaw angle relative to path | [rad]
        :param t: time | [s]
        """
        self._values = np.array([V,s,ey,epsi,t])
        self._keys = ['V','s','ey','epsi','t']
        self._syms = ca.vertcat(*[ca.SX.sym(self._keys[i]) for i in range(len(self._keys))])
    
    @property
    def V(self): return self.values[0] 
    
    @property
    def s(self): return self.values[1]
    
    @property
    def ey(self): return self.values[2]
    
    @ey.setter
    def ey(self, value): self.values[2] = value
    
    @property
    def epsi(self): return self.values[3]
    
    @epsi.setter
    def epsi(self, value): self.values[3] = value
    
    @property
    def t(self): return self.values[4]
    
    @property
    def values(self): return self._values
    
    @property
    def syms(self): return self._syms
    
    @property
    def keys(self): return self._keys
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
