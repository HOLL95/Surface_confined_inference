"""
Options management system with inheritance support.
This module provides the metaclass and base class for option configuration.
"""
from typing import Any, Dict, List, Set, Type, ClassVar
import inspect
from ._OptionsDescriptor import OptionDescriptor


class OptionsMeta(type):
    """
    Metaclass that handles options inheritance and registration.

    This metaclass automatically manages option descriptors across class hierarchies,
    ensuring that subclasses inherit options from their parent classes while also
    adding their own specific options. It maintains a registry of all available
    options for each class.

    The metaclass:
    - Collects option descriptors from parent classes
    - Registers new option descriptors defined in the current class
    - Maintains a `_options` set containing all available option names
    - Enables proper inheritance of validation and default behaviors

    Example:
        class BaseOptions(OptionsManager):
            base_param = NumberOption("base_param", default=1.0)

        class SpecificOptions(BaseOptions):
            specific_param = StringOption("specific_param", default="test")

        # SpecificOptions._options will contain both "base_param" and "specific_param"
    """

    def __new__(mcs, name, bases, attrs):
        """
        Create a new options class with inherited and new option descriptors.

        Args:
            mcs: The metaclass itself
            name (str): Name of the class being created
            bases (tuple): Tuple of base classes
            attrs (dict): Class attributes dictionary

        Returns:
            type: The newly created class with properly registered options
        """
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
    """
    Base class for managing options with inheritance support.

    This class provides the foundation for all options management in the system.
    It uses the OptionsMeta metaclass to automatically handle option inheritance
    and provides methods for option manipulation, validation, and serialization.

    Key features:
    - Automatic option discovery and inheritance
    - Default value initialization from descriptors
    - Option validation on assignment
    - Dictionary-style option access and updates
    - Unknown option handling with customizable behavior

    Attributes:
        _options (ClassVar[Set[str]]): Set of all registered option names for this class

    Example:
        class MyOptions(OptionsManager):
            temperature = NumberOption("temperature", default=298.0)
            method = StringOption("method", default="simulation")

        # Create instance with defaults
        options = MyOptions()

        # Create instance with custom values
        options = MyOptions(temperature=300.0, method="optimization")

        # Access as dictionary
        opt_dict = options.as_dict()

        # Update multiple options
        options.update(temperature=295.0, method="analysis")
    """
    
    # Class variable to store all registered option names
    _options: ClassVar[Set[str]] = set()
    
    def __init__(self, **kwargs):
        """
        Initialize options with provided values or defaults.

        This method sets up the options instance by applying default values
        from option descriptors and then overriding with any provided values.
        Unknown options are handled according to the class's policy.

        Args:
            **kwargs: Keyword arguments for option values. Keys must match
                     registered option names or will trigger unknown option handling.

        Raises:
            AttributeError: If unknown options are provided and not handled
            ValueError: If option values fail descriptor validation
            TypeError: If option values are of incorrect types

        Example:
            # Initialize with defaults
            options = MyOptions()

            # Initialize with custom values
            options = MyOptions(temperature=300.0, method="optimization")
        """
        # Apply default values from descriptors
        for option_name in self._options:
            descriptor = self.__class__.__dict__.get(option_name)
            if option_name in kwargs:
                setattr(self, option_name, kwargs[option_name])
            elif descriptor and isinstance(descriptor, OptionDescriptor):
                setattr(self, option_name, descriptor.default)

        for key, value in kwargs.items():
            if key not in self._options:
                self._handle_unknown_option(key, value)
    
    def _handle_unknown_option(self, key: str, value: Any) -> None:
        """Handle an unknown option. Subclasses can override this."""
        raise AttributeError(f"Unknown option: {key}")
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute with validation if it's a registered option."""
        if name in self._options:
            # Possibly trigger validation via descriptor, etc.
            super().__setattr__(name, value)
        else:
            if not name.startswith('_'):
                self._handle_unknown_attribute(name, value)
            else:
                super().__setattr__(name, value)
    
    def _handle_unknown_attribute(self, name: str, value: Any) -> None:
        """Handle setting an unknown attribute. Subclasses can override this."""
        # Default behavior: warn, but still allow
        #print(f"Warning: Unknown attribute '{name}' set.")
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

