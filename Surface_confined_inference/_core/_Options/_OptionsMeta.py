"""
Options management system with inheritance support.
This module provides the metaclass and base class for option configuration.
"""
from typing import Any, Dict, List, Set, Type, ClassVar
import inspect
from ._OptionsDescriptor import OptionDescriptor


class OptionsMeta(type):
    """Metaclass that handles options inheritance and registration."""
    
    def __new__(mcs, name, bases, attrs):
        # Create a set to hold all the option names for this class
        options_set = set()
        
        # Collect options from base classes
        for base in bases:
            if hasattr(base, '_options'):
                options_set.update(base._options)
        
        # Add options from this class
        for key, value in attrs.items():
            if isinstance(value, OptionDescriptor):
                options_set.add(key)
        
        # Store the options set in the class
        attrs['_options'] = options_set
        #print(f"Creating {name} with options: {options_set}")
        
        return super().__new__(mcs, name, bases, attrs)


class OptionsManager(metaclass=OptionsMeta):
    """Base class for managing options with inheritance support."""
    
    # Class variable to store all registered option names
    _options: ClassVar[Set[str]] = set()
    
    def __init__(self, **kwargs):
        """Initialize options with provided values or defaults."""
        # Apply default values from descriptors
        for option_name in self._options:
            descriptor = self.__class__.__dict__.get(option_name)
            if descriptor and isinstance(descriptor, OptionDescriptor):
                setattr(self, option_name, descriptor.default)
        
        # Override with any provided values
        for key, value in kwargs.items():
            if key in self._options:
                setattr(self, key, value)
            else:
                self._handle_unknown_option(key, value)
    
    def _handle_unknown_option(self, key: str, value: Any) -> None:
        """Handle an unknown option. Subclasses can override this."""
        raise AttributeError(f"Unknown option: {key}")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute with validation if it's an option."""
        if name in self._options:
            # Let the descriptor handle validation
            super().__setattr__(name, value)
        else:
            # For non-options, check if we should warn or allow
            if name.startswith('_'):
                # Private attributes are always allowed
                super().__setattr__(name, value)
            else:
                # Let subclasses decide how to handle this
                self._handle_unknown_attribute(name, value)
    
    def _handle_unknown_attribute(self, name: str, value: Any) -> None:
        """Handle setting an unknown attribute. Subclasses can override this."""
        # By default, just set the attribute
        super().__setattr__(name, value)
    
    def as_dict(self) -> Dict[str, Any]:
        """Return all options as a dictionary."""
        return {name: getattr(self, name) for name in self._options}
    
    def update(self, **kwargs) -> None:
        """Update multiple options at once."""
        for key, value in kwargs.items():
            if key in self._options:
                setattr(self, key, value)
            else:
                self._handle_unknown_option(key, value)
    
    @classmethod
    def get_option_names(cls) -> List[str]:
        """Get all option names for this class."""
        return list(cls._options)
    
    @classmethod
    def get_option_defaults(cls) -> Dict[str, Any]:
        """Get all default option values for this class."""
        defaults = {}
        for name in cls._options:
            descriptor = cls.__dict__.get(name)
            if descriptor and isinstance(descriptor, OptionDescriptor):
                defaults[name] = descriptor.default
        return defaults


class WarningOptionsManager(OptionsManager):
    """An options manager that warns about unknown options but allows them."""
    
    def __init__(self, **kwargs):
        self._warned_attrs = set()
        super().__init__(**kwargs)
    
    
