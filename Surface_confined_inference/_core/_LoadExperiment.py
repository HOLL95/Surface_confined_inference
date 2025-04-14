import numpy as np
import json
import Surface_confined_inference as sci
import json

class LoadExperiment:
    _registry = {}
    
    @classmethod
    def register(cls, experiment_type=None):
        """Decorator to register experiment classes"""
        def decorator(subclass):
            name = experiment_type or subclass.__name__.lower()
            cls._registry[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def get_class(cls, experiment_type):
        """Get experiment class by type name"""
        if experiment_type not in cls._registry:
            raise KeyError(f"Unknown experiment type: {experiment_type}. Available types: {list(cls._registry.keys())}")
        return cls._registry[experiment_type]
        
