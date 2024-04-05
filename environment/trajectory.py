from abc import ABC, abstractmethod
import numpy as np

class Trajectory(ABC):
    
    @abstractmethod
    def update(self, t: float):
        '''
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                p_d,     position, m
                pd_d,    velocity, m/s
                pdd_d,   acceleration, m/s**2
        '''
        pass

class Circle(Trajectory):
    def __init__(self, T = 6, center=np.array([0,0]), radius=1, freq=0.2):
        """
        This is the constructor for the circle trajectory

        Inputs:
            center, the center of the circle (m)
            radius, the radius of the circle (m)
            freq, the frequency with which a circle is completed (Hz)
        """
        self.center = center
        self.cx, self.cy = center[0], center[1]
        self.radius = radius
        self.freq = freq
        self.omega = 2*np.pi*self.freq
        self.T = T
        
    def update(self, t):
        p     = np.array([self.cx + self.radius*np.cos(self.omega*t),
                            self.cy + self.radius*np.sin(self.omega*t)])
        pd    = np.array([-self.radius*self.omega*np.sin(self.omega*t),
                            self.radius*self.omega*np.cos(self.omega*t)])
        pdd   = np.array([-self.radius*((self.omega)**2)*np.cos(self.omega*t),
                            -self.radius*((self.omega)**2)*np.sin(self.omega*t)])
        return {'p': p, 'pd':pd , 'pdd':pdd}
