"""
Options specific to electrochemical experiments.
This module provides option classes for different types of electrochemical experiments.
"""
import collections.abc
import numbers
from typing import List, Optional, Sequence

from ._OptionsDescriptor import (
    BoolOption, EnumOption, NumberOption, SequenceOption, StringOption, OptionDescriptor
)
from ._OptionsMeta import OptionsManager




class BaseExperimentOptions(OptionsManager):
    """Base options common to all electrochemical experiments."""
    
    experiment_type = EnumOption(
        "experiment_type",
        allowed_values=["FTACV", "PSV", "DCV", "SquareWave"],
        default=None,
        doc="Type of experiment (FTACV, PSV, DCV, SquareWave)."
    )
    
    GH_quadrature = BoolOption(
        "GH_quadrature",
        default=True,
        doc="Whether to implement Gauss-Hermite quadrature for approximating normal distributions in dispersion."
    )
    
    
    normalise_parameters = BoolOption(
        "normalise_parameters",
        default=False,
        doc="In CMAES, it is convenient to normalise the parameters to between 0 and 1 when searching in parameter space."
    )
    
    kinetics = EnumOption(
        "kinetics",
        allowed_values=["ButlerVolmer", "Marcus", "Nernst"],
        default="ButlerVolmer",
        doc="Type of electrochemical kinetics to use."
    )
    
    dispersion = BoolOption(
        "dispersion",
        default=False,
        doc="Whether to model dispersion in parameters."
    )
    
    dispersion_bins = SequenceOption(
        "dispersion_bins",
        default=[], 
        item_type=int,
        doc="Number of bins used to approximate each dispersion distribution."
    )
    
    dispersion_test = BoolOption(
        "dispersion_test",
        default=False,
        doc="Defines whether to save the unweighted individual simulations of a dispersed simulation."
    )
    
    transient_removal = NumberOption(
        "transient_removal",
        default=0,
        doc="Amount of time to remove from the beginning of the simulation to eliminate transient effects."
    )
    
    problem = EnumOption(
        "problem",
        allowed_values=["forwards", "inverse"],
        default="forwards",
        doc="Defines whether the problem is forwards (simulation) or inverse (parameter estimation)."
    )
    
    Faradaic_only = BoolOption(
        "Faradaic_only",
        default=False,
        doc="Whether to return only the Faradaic component of the current."
    )


class FTACVOptions(BaseExperimentOptions):
    """Options specific to FTACV experiments."""
    
    def __init__(self, **kwargs):
        # Set default experiment type
        kwargs.setdefault("experiment_type", "FTACV")
        super().__init__(**kwargs)
    
    Fourier_fitting = BoolOption(
        "Fourier_fitting",
        default=False,
        doc="Used (in combination with the appropriate likelihood function) to fit Fourier spectrum data."
    )
    
    Fourier_function = EnumOption(
        "Fourier_function",
        allowed_values=["composite", "abs", "real", "imaginary", "inverse"],
        default="abs",
        doc="Defines how to represent Fourier filtered current data."
    )
    
    Fourier_harmonics = SequenceOption(
        "Fourier_harmonics",
        default=list(range(0, 10)),
        item_type=int,
        doc="Defines the harmonics to be included in the filtered Fourier spectrum."
    )
    
    Fourier_window = EnumOption(
        "Fourier_window",
        allowed_values=["hanning", False],
        default="hanning",
        doc="Defines whether or not to use a windowing function when applying Fourier filtration methods."
    )
    
    top_hat_width = NumberOption(
        "top_hat_width",
        default=0.5,
        doc="Defines the width of the top hat window (as a percentage of the input frequency) around which to extract the individual harmonics."
    )
    phase_only = BoolOption(
        "phase_only",
        default=True,
        doc="Whether to fit the phase of the capacitance current as the same value as that of the phase of the Faradaic current."
    )


class PSVOptions(FTACVOptions):
    """Options specific to PSV experiments."""
    
    def __init__(self, **kwargs):
        # Set default experiment type
        kwargs.setdefault("experiment_type", "PSV")
        super().__init__(**kwargs)
    
    # Add PSV-specific options here
    PSV_num_peaks = NumberOption(
        "PSV_num_peaks",
        default=50,
        min_value=1,
        doc="Number of peaks to simulate in PSV."
    )


class DCVOptions(BaseExperimentOptions):
    """Options specific to DCV experiments."""
    
    def __init__(self, **kwargs):
        # Set default experiment type
        kwargs.setdefault("experiment_type", "DCV")
        super().__init__(**kwargs)
    
    # Add DCV-specific options here


class SquareWaveOptions(BaseExperimentOptions):
    """Options specific to SquareWave experiments."""
    
    def __init__(self, **kwargs):
        # Set default experiment type
        kwargs.setdefault("experiment_type", "SquareWave")
        super().__init__(**kwargs)
    
    # Add SquareWave-specific options here
    square_wave_return = EnumOption(
        "square_wave_return",
        allowed_values=["forwards", "backwards", "net", "total"],
        default="net",
        doc="Which component of the square wave response to return."
    )


class SingleExperimentOptions(BaseExperimentOptions):
    _experiment_classes = {
        "FTACV": FTACVOptions,
        "PSV": PSVOptions,
        "DCV": DCVOptions,
        "SquareWave": SquareWaveOptions
    }
    def __init__(self,options_handler=None,**kwargs):
        experiment_type=kwargs["experiment_type"]
        if experiment_type not in self._experiment_classes:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
        
        base_cls = self._experiment_classes[experiment_type]

        if options_handler is None or options_handler is base_cls:
            # No custom handler: use base directly
            self._experiment_options = base_cls(**kwargs)
        else:
            # Dynamically create a new class that inherits from both
            # Custom comes first so it overrides base where needed
            CombinedOptions = type(
            f"Combined{options_handler.__name__}{base_cls.__name__}",
                (options_handler, base_cls),
                {}
            )
           
            self._experiment_options = CombinedOptions(**kwargs)


        # Copy values from experiment options into self, if names match
        
        for name in self._experiment_options.get_option_names():
            
            setattr(self, name, getattr(self._experiment_options, name))

        super().__init__(**kwargs)

    def as_dict(self):
        """Return a merged dictionary of all options (self + experiment options)."""
        combined = self.experiment_options.as_dict()
        combined.update(super().as_dict())
        return combined

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Dynamically attach properties from all possible experiment types."""
        super().__init_subclass__(**kwargs)
        descriptors_added = set()

        for exp_cls in cls._experiment_classes.values():
            for name in exp_cls.get_option_names():
                if name not in descriptors_added:
                    descriptor = exp_cls.__dict__.get(name)
                    if descriptor and isinstance(descriptor, OptionDescriptor):
                        setattr(cls, name, descriptor)
                        descriptors_added.add(name)
                        
