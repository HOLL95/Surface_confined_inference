from abc import ABC, abstractmethod
class OptionsAwareMixin:
    _core_options=["boundaries", "fixed_parameters", "optim_list"]
    def __setattr__(self, name, value):
       
        if name == "OptionsDecorator":
            super().__setattr__(name, value)
            return

        # Only if _internal_options is defined and active
        if hasattr(self, "_internal_options") and hasattr(self._internal_options, "get_option_names"):
            
            valid_options = self._internal_options.get_option_names()
            if name in valid_options:
                setattr(self._internal_options, name, value)
                super().__setattr__(name, value)
                return

        # Warn about unknown attributes
        if not name.startswith("_") and name not in self._core_options:
            print(f"Warning: '{name}' is not in the list of accepted options and will not affect simulation behavior")
        super().__setattr__(name, value)