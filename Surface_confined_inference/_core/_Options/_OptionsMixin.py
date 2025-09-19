"""
Options mixin class for electrochemical experiment configuration.

This module provides the OptionsAwareMixin class that enables automatic validation
and management of experiment options. Classes that inherit from this mixin can
define options using descriptors and have them automatically validated when set.

The mixin integrates with the options management system to provide:
- Automatic option validation through descriptors
- Warning system for unknown options
- Support for manual options that bypass validation
- Core options that are always accepted
"""
from abc import ABC, abstractmethod

class OptionsAwareMixin:
    """
    Mixin class that provides automatic options validation and management.

    This mixin enables classes to automatically validate options when they are set,
    providing a clean interface between user input and internal option management.
    It works in conjunction with options managers that define validation rules
    through descriptors.

    Attributes:
        _core_options (list): List of core option names that are always accepted.
            These options are fundamental to the class and bypass normal validation.

    The mixin supports three types of options:
    1. Validated options: Defined in _internal_options, validated by descriptors
    2. Manual options: Listed in _manual_options, bypass validation
    3. Core options: Always accepted, no validation
    """
    _core_options=["boundaries", "fixed_parameters", "optim_list"]

    def __setattr__(self, name, value):
        """
        Override attribute setting to provide automatic options validation.

        This method intercepts all attribute assignments and routes them through
        the appropriate validation system based on the attribute name and type.

        Args:
            name (str): The name of the attribute being set
            value: The value to assign to the attribute

        Behavior:
            1. If the attribute is a validated option (in _internal_options),
               it validates the value using the option's descriptor and sets
               it both in the internal options manager and on the instance.

            2. If the attribute is a manual option (in _manual_options),
               it sets the value directly without validation.

            3. If the attribute is a core option (in _core_options),
               it sets the value directly without validation.

            4. For unknown attributes that don't start with underscore,
               it prints a warning but still sets the value.

            5. All other attributes (including private ones starting with _)
               are set normally without validation or warnings.

        Warning:
            Unknown public attributes will trigger a warning message indicating
            they won't affect simulation behavior, helping users identify
            potential typos in option names.

        """
        # Only if _internal_options is defined and active
        if hasattr(self, "_internal_options") and hasattr(self._internal_options, "get_option_names"):

            valid_options = self._internal_options.get_option_names()
            if name in valid_options:
                setattr(self._internal_options, name, value)
                super().__setattr__(name, value)
                return
        # Check if the attribute is in the manual options list
        if hasattr(self, "_manual_options") and name in self._manual_options:
            super().__setattr__(name, value)
            return
        # Warn about unknown attributes
        if not name.startswith("_") and name not in self._core_options:
            print("Warning: '{0}' is not in the list of accepted options for `{1}` and will not affect simulation behavior".format(name, self.__class__.__name__))
        super().__setattr__(name, value)