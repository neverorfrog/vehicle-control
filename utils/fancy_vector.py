from typing import Union
from abc import abstractmethod
from abc import ABC
import numpy as np

class FancyVector(ABC):
    """
    Abstract Base Class for States and Actions
    In general it holds values and casadi variables (that's why it's fancy)
    """
    
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    
    @property
    def values(self) -> np.ndarray: return self._values #redefined by subclass
    
    @property
    def keys(self) -> list: return self._keys #redefined by subclass
    
    @property
    def syms(self): return self._syms #redefined by subclass
    
    @property
    def labels(self): return self._labels #redefined by subclass
    
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
        return str({key: f"{value:.2f}" for key,value in dict(zip(self.keys, self.values)).items()})
    
    def __add__(self, other):
        """
        Overload + operator.
        :param other: numpy array to be added to state vector
        """
        assert isinstance(other, (self.__class__)), "You can only sum two same states"
        tobesummed = other.values if isinstance(other, FancyVector) else other
        new_state = self.values + tobesummed
        return self.__class__.create(*new_state)