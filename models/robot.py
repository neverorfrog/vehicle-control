from abc import abstractmethod
from abc import ABC
from matplotlib.axes import Axes
from utils.fancy_vector import FancyVector

class Robot(ABC):
    def __init__(self, config: dict):
        # Set sampling time
        self.dt = config['dt']
        # Configuration
        self.config = config
        # Initialize state
        self.state: FancyVector = self.__class__.create_state()
        # Initialize input
        self.input: FancyVector = self.__class__.create_action()
        # Initialize ode 
        self._init_model()
    
    @abstractmethod
    def _init_model(self):
        '''Initializes the casadi transition function responsible for evolving the state of the robot''' 
        pass
    
    @abstractmethod
    def plot(self, axis: Axes, state): pass
    
    @property
    @abstractmethod
    def transition(self): pass
    
    @classmethod
    @abstractmethod
    def create_state(cls, *args, **kwargs) -> FancyVector:
        pass
    
    @classmethod
    @abstractmethod
    def create_action(cls, *args, **kwargs) -> FancyVector:
        pass
    
    def integrate(self,state,action,ode,h):
        '''
        RK4 integrator
        h: integration interval
        '''
        state_dot_1 = ode(state, action)
        state_1 = state + (h/2)*state_dot_1
        
        state_dot_2 = ode(state_1, action)
        state_2 = state + (h/2)*state_dot_2
        
        state_dot_3 = ode(state_2, action)
        state_3 = state + h*state_dot_3
        
        state_dot_4 = ode(state_3, action)
        state = state + (1/6) * (state_dot_1 + 2 * state_dot_2 + 2 * state_dot_3 + state_dot_4) * h
        
        return state