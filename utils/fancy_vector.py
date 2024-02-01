from typing import Union
from abc import abstractmethod
from abc import ABC

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