"""
Descriptor classes for option validation in the options management system.
These descriptors handle type checking and validation for different option types.
"""
import collections.abc
import numbers
from typing import Any, Dict, List, Optional, Type, Union, Sequence


class OptionDescriptor:
    """Base descriptor class for options with validation."""
    
    def __init__(self, name: str, default: Any = None, doc: str = None):
        self.name = name
        self.private_name = f"_{name}"
        self.default = default
        self.__doc__ = doc
    
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name, self.default)
    
    def __set__(self, instance, value):
        #print(self.private_name, "called", value)
        self.validate(value)
        setattr(instance, self.private_name, value)
    
    def validate(self, value: Any) -> None:
        print("Not passed")
        """Validate the value. Override in subclasses."""
        pass


class TypedOption(OptionDescriptor):
    """Descriptor for options that must match specific types."""
    
    def __init__(self, name: str, allowed_types: Union[Type, List[Type]], default: Any = None, doc: str = None):
        super().__init__(name, default, doc)
        if not isinstance(allowed_types, (list, tuple)):
            allowed_types = [allowed_types]
        self.allowed_types = allowed_types
    
    def validate(self, value: Any) -> None:
        """Validate that the value is of the allowed types."""
        if not any(isinstance(value, t) for t in self.allowed_types):
            type_names = [t.__name__ for t in self.allowed_types]
            type_str = " or ".join(type_names)
            raise TypeError(f"{self.name} must be of type {type_str}, got {type(value).__name__}")


class EnumOption(OptionDescriptor):
    """Descriptor for options that must be one of a set of allowed values."""
    
    def __init__(self, name: str, allowed_values: List[Any], default: Any = None, doc: str = None):
        super().__init__(name, default, doc)
        self.allowed_values = allowed_values
    
    def validate(self, value: Any) -> None:
        """Validate that the value is one of the allowed values."""
        if value not in self.allowed_values:
            values_str = ", ".join(repr(v) for v in self.allowed_values)
            raise ValueError(f"{self.name} must be one of: {values_str}, got {repr(value)}")


class BoolOption(TypedOption):
    """Descriptor for boolean options."""
    
    def __init__(self, name: str, default: bool = False, doc: str = None):
        super().__init__(name, bool, default, doc)


class NumberOption(TypedOption):
    """Descriptor for numeric options."""
    
    def __init__(self, name: str, default: numbers.Number = 0, 
                 min_value: Optional[numbers.Number] = None,
                 max_value: Optional[numbers.Number] = None,
                 doc: str = None):
        super().__init__(name, numbers.Number, default, doc)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> None:
        """Validate that the value is a number within the specified range."""
        super().validate(value)
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be at least {self.min_value}, got {value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be at most {self.max_value}, got {value}")


class SequenceOption(TypedOption):
    """Descriptor for sequence options."""
    
    def __init__(self, name: str, default: Sequence = None, 
                 item_type: Optional[Type] = None,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 doc: str = None):
        super().__init__(name, collections.abc.Sequence, default or [], doc)
        self.item_type = item_type
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any) -> None:
        """Validate that the value is a sequence with optional constraints."""
        super().validate(value)
        if self.min_length is not None and len(value) < self.min_length:
            raise ValueError(f"{self.name} must have at least {self.min_length} items, got {len(value)}")
        if self.max_length is not None and len(value) > self.max_length:
            raise ValueError(f"{self.name} must have at most {self.max_length} items, got {len(value)}")
       
        if self.item_type is not None:
            for i, item in enumerate(value):
               
                if not isinstance(item, self.item_type):
                    raise TypeError(f"Item {i} in {self.name} must be of type {self.item_type.__name__}, "
                                    f"got {type(item).__name__}")


class StringOption(TypedOption):
    """Descriptor for string options."""
    
    def __init__(self, name: str, default: str = "", 
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 doc: str = None):
        super().__init__(name, str, default, doc)
        self.min_length = min_length
        self.max_length = max_length
    
    def validate(self, value: Any) -> None:
        """Validate that the value is a string within the specified length range."""
        super().validate(value)
        if self.min_length is not None and len(value) < self.min_length:
            raise ValueError(f"{self.name} must have at least {self.min_length} characters, got {len(value)}")
        if self.max_length is not None and len(value) > self.max_length:
            raise ValueError(f"{self.name} must have at most {self.max_length} characters, got {len(value)}")


class DictOption(TypedOption):
    """Descriptor for dictionary options."""
    
    def __init__(self, name: str, default: Dict = None,
                 key_type: Optional[Type] = None,
                 value_type: Optional[Type] = None,
                 required_keys: Optional[List[Any]] = None,
                 doc: str = None):
        super().__init__(name, dict, default or {}, doc)
        self.key_type = key_type
        self.value_type = value_type
        self.required_keys = required_keys or []
    
    def validate(self, value: Any) -> None:
        """Validate that the value is a dictionary with optional constraints."""
        super().validate(value)
        if self.required_keys:
            missing_keys = [key for key in self.required_keys if key not in value]
            if missing_keys:
                missing_str = ", ".join(repr(k) for k in missing_keys)
                raise ValueError(f"{self.name} is missing required keys: {missing_str}")
        
        if self.key_type or self.value_type:
            for k, v in value.items():
                if self.key_type and not isinstance(k, self.key_type):
                    raise TypeError(f"Key {repr(k)} in {self.name} must be of type {self.key_type.__name__}, "
                                    f"got {type(k).__name__}")
                if self.value_type and not isinstance(v, self.value_type):
                    raise TypeError(f"Value for key {repr(k)} in {self.name} must be of type "
                                    f"{self.value_type.__name__}, got {type(v).__name__}")