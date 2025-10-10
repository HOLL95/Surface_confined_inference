"""
Descriptor classes for option validation in the options management system.
These descriptors handle type checking and validation for different option types.
"""
import collections.abc
import numbers
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, Union


class OptionDescriptor:
    """Base descriptor class for options with validation."""
    
    def __init__(self, name: str, default: Any = None, doc: str = None, required: bool =False):
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
            type_str = ' or '.join(type_names)
            raise TypeError(f"{self.name} must be of type {type_str}, got {type(value).__name__}")


class EnumOption(OptionDescriptor):
    """Descriptor for options that must be one of a set of allowed values."""
    
    def __init__(self, name: str, allowed_values: List[Any], default: Any = None, doc: str = None):
        super().__init__(name, default, doc)
        self.allowed_values = allowed_values
    
    def validate(self, value: Any) -> None:
        """Validate that the value is one of the allowed values."""
        if value not in self.allowed_values:
            values_str = ', '.join(repr(v) for v in self.allowed_values)
            raise ValueError(f"{self.name} must be one of: {values_str}, got {value!r}")


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
class ExclusiveSequenceOption(SequenceOption):
    """Descriptor for sequence options where all values must be present."""
    def __init__(self, name: str, default: Sequence = None, 
                    target: Optional[Sequence]=[],
                    doc: str = None):
                    super().__init__(name, collections.abc.Sequence, default or [], doc)
                    self.target=set(target)


    def validate(self, value: Any) -> None:
        super().validate(value)
        if value is not None:
            value_set=set(value)
            missing=self.target.difference(value_set)
            if len(missing)>0:
                raise ValueError(f"In option {self.name} the following values are missing {' '.join(list(missing))}")
            present=value_set.difference(self.target)
            if len(present)>0:
                raise ValueError(f"In option {self.name} the following values should not be present {' '.join(list(present))}")


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
                missing_str = ', '.join(repr(k) for k in missing_keys)
                raise ValueError(f"{self.name} is missing required keys: {missing_str}")
        
        if self.key_type or self.value_type:
            for k, v in value.items():
                if self.key_type and not isinstance(k, self.key_type):
                    raise TypeError(f"Key {k!r} in {self.name} must be of type {self.key_type.__name__}, "
                                    f"got {type(k).__name__}")
                if self.value_type and not isinstance(v, self.value_type):
                    raise TypeError(f"Value for key {k!r} in {self.name} must be of type "
                                    f"{self.value_type.__name__}, got {type(v).__name__}")
class ExclusiveDictOption(DictOption):
    """Descriptor for sequence options where all values must be present."""
    def __init__(self, name: str, default: Dict = None, 
                target: Optional[Sequence]=[],
                value_type: Optional[Type] = None,
                doc: str = None):
                super().__init__(name, default=default or {}, key_type=None,value_type=value_type,required_keys=None,doc=doc)
                self.target=set(target)
                self.value_type=value_type

    def validate(self, value: Any) -> None:
        super().validate(value)
        if value is not None:
            value_set=set(value.keys())
            missing=self.target.difference(value_set)
            if len(missing)>0:
                raise ValueError(f"In option {self.name} the following values are missing {' '.join(list(missing))}")
            present=value_set.difference(self.target)
            if len(present)>0:
                raise ValueError(f"In option {self.name} the following values should not be present {' '.join(list(present))}")
class ComposedOption(OptionDescriptor):
    """Descriptor for options where each item in a sequence must be validated by at least one of the provided validators."""
    
    def __init__(self, name: str, validators: List[OptionDescriptor], default: Sequence = None, doc: str = None):
        """
        Initialize a ComposedOption with a list of validator options.
        
        Args:
            name: Name of the option
            validators: List of OptionDescriptor instances to validate against
            default: Default value (a sequence)
            doc: Documentation string
        """
        super().__init__(name, default or [], doc)
        self.name
        self.validators = validators
    
    def validate(self, value: Any) -> None:
        """
        Validate that each item in the sequence is valid according to at least one validator.
        
        Raises:
            TypeError: If value is not a sequence
            ValueError: If any item fails validation by all validators
        """
        if not isinstance(value, collections.abc.Sequence):
            raise TypeError(f"{self.name} must be a sequence, got {type(value).__name__}")
        
        for i, item in enumerate(value):
            valid = False
            validation_errors = []
            
            for validator in self.validators:
                try:
                    # Create a temporary descriptor instance with the same name
                    temp_validator = validator(f"{self.name}_{i}")
                    temp_validator.validate(item)
                    valid = True
                    break
                except (TypeError, ValueError) as e:
                    validation_errors.append(str(e))
            
            if not valid:
                error_msg = ' '.join(validation_errors)
                raise ValueError(f"Item {i} in {self.name} failed validation with all validators:\n{error_msg}")


class FileOption(TypedOption):
    """Descriptor for file path options with existence checking."""
    
    def __init__(self, name: str, default: str = "", 
                 must_exist: bool = True, 
                 check_readable: bool = False,
                 check_writable: bool = False,
                 doc: str = None):
        """
        Initialize a FileOption.
        
        Args:
            name: Name of the option
            default: Default file path
            must_exist: Whether the file must exist
            doc: Documentation string
        """
        super().__init__(name, str, default, doc)
        self.must_exist = must_exist
    
    def validate(self, value: Any) -> None:
        """
        Validate that the value is a string representing a valid file path.
        
        Raises:
            TypeError: If value is not a string
            ValueError: If file doesn't exist or permissions checks fail
        """
        super().validate(value)
        
        if not value:  # Allow empty string if not requiring existence
            if self.must_exist:
                raise ValueError(f"{self.name} cannot be empty when must_exist=True")
            return
        
        if self.must_exist and not os.path.isfile(value):
            raise ValueError(f"{self.name} must be a path to an existing file, got '{value}'")

class DirectoryOption(TypedOption):
    """Descriptor for directory path options with existence and emptiness checking."""
    
    def __init__(self, name: str, default: str = "", 
                 must_exist: bool = True,
                 must_be_empty: bool = False,
                 can_create=False,
                 doc: str = None):
        """
        Initialize a DirectoryOption.
        
        Args:
            name: Name of the option
            default: Default directory path
            must_exist: Whether the directory must exist
            must_be_empty: Whether the directory must be empty
            doc: Documentation string
        """
        super().__init__(name, str, default, doc)
        self.must_exist = must_exist
        self.must_be_empty = must_be_empty
        self.can_create=can_create
    
    def validate(self, value: Any) -> None:
        """
        Validate that the value is a string representing a valid directory path.
        
        Raises:
            TypeError: If value is not a string
            ValueError: If directory doesn't exist, isn't empty, or permissions checks fail
        """
        super().validate(value)
        
        if not value:  # Allow empty string if not requiring existence
            if self.must_exist:
                raise ValueError(f"{self.name} cannot be empty when must_exist=True")
            return
        
        if self.must_exist and not os.path.isdir(value):
            if self.can_create==True:
                Path(value).mkdir(parents=True, exist_ok=False)
            else:
                raise ValueError(f"{self.name} must be a path to an existing directory, got '{value}'")
            
        if self.must_exist and self.must_be_empty:
            # Check if directory is empty (only if it exists)
            if os.path.isdir(value) and os.listdir(value):
                raise ValueError(f"{self.name} directory '{value}' must be empty")