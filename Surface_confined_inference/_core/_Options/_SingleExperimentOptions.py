"""
Options specific to electrochemical experiments.
This module provides option classes for different types of electrochemical experiments.
"""
import numbers
import os
from sympy import Basic

from ._OptionsDescriptor import (
    BoolOption,
    EnumOption,
    DictOption,
    ExclusiveDictOption,
    NumberOption,
    OptionDescriptor,
    SequenceOption,
    TypedOption,
    NoneOption, 
    AnyOfOption,
    FileOption,
)
from ._OptionsMeta import OptionsManager, OptionsMeta


class BaseExperimentOptions(OptionsManager):
    """Base options common to all electrochemical experiments."""
    
    experiment_type = EnumOption(
        "experiment_type",
        allowed_values=["FTACV", "PSV", "DCV", "SquareWave", "Generic", "SWVtanh"],
        default=None,
        doc="Type of experiment (FTACV, PSV, DCV, SquareWave, Generic)."
    )
    model=EnumOption(
        "model",
        allowed_values=["single_electron", "square_scheme"],
        default="single_electron",
        doc="Built-in C++ model for the system, either a single electron redox reaction "
            "or a 3x3 square scheme. Ignored when `mechanism` is set, since the ODE "
            "system is then generated from the reaction network instead."
    )
    mechanism = AnyOfOption(
        "mechanism",
        validators=[FileOption, DictOption, NoneOption],
        default=None,
        doc="Reaction network to generate the ODE system from: a path to a YAML file, "
            "a YAML document as a string, or an already-parsed mapping. Setting this is "
            "what selects the generated model over the built-in `model`. save_class "
            "always writes the parsed mapping, since a path does not travel between "
            "machines."
    )
    poly_degree = NumberOption(
        "poly_degree",
        default=15,
        min_value=2,
        doc="Number of coefficients in the polynomial expansion of log(k) in "
            "overpotential, for model='mechanism'. Butler-Volmer is exact at 2."
    )
    diffsl_solver = EnumOption(
        "diffsl_solver",
        allowed_values=["tsit45", "bdf", "tr_bdf2", "esdirk34"],
        default="tsit45",
        doc="Solver used for model='mechanism'."
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
        allowed_values=["ButlerVolmer", "Marcus"],
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
    parallel_cpu = NumberOption(
        "Parallel_cpu",
        default=len(os.sched_getaffinity(0)),
        doc="Number of CPUs for parallel simulations for dispersion"
    )

    @property
    def effective_model(self):
        """Which model will actually be simulated: `mechanism` wins over `model`.

        Read-only, and not an option itself, so it stays out of `as_dict` and
        out of the saved file.
        """
        if self.mechanism is not None:
            return "mechanism"
        return self.model




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
    input_params=ExclusiveDictOption(
        "input_params",
        value_type=numbers.Number,
        target=["E_start", "E_reverse","v", "omega", "phase", "delta_E",]+["area", "Temp",  "Surface_coverage"],
        default={},
        doc="Necessary input params for FTACV"
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
    input_params=ExclusiveDictOption(
        "input_params",
        value_type=numbers.Number,
        target=["Edc","omega", "phase", "delta_E"]+["area", "Temp",  "Surface_coverage"],
        doc="Necessary input params for PSV"
    )

class DCVOptions(BaseExperimentOptions):
    """Options specific to DCV experiments."""
    
    def __init__(self, **kwargs):
        # Set default experiment type
        kwargs.setdefault("experiment_type", "DCV")
        super().__init__(**kwargs)
    input_params=ExclusiveDictOption(
        "input_params",
        value_type=numbers.Number,
        target=["E_start", "E_reverse","v"]+["area", "Temp", "Surface_coverage"],
        doc="Necessary input params for DCV"
    )
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
    input_params=ExclusiveDictOption(
        "input_params",
        value_type=numbers.Number,
        target=[
                "omega",
                "Estart",
                "Estop",
                "Eamp",
                "Estep",
                "sampling_factor",
            ]+["area", "Temp",  "Surface_coverage"],
        doc="Necessary input params for SWV. Estop is the potential the staircase "
            "ends at, so the scan direction is the sign of Estop-Estart. Older "
            "dictionaries using E_start/delta_E/scan_increment/SW_amplitude/v are "
            "converted, with a DeprecationWarning, by "
            "`convert_legacy_square_wave_params`."
    )
class SquareWavetanhOptions(SquareWaveOptions):
    """Options specific to tanh-smoothed SquareWave experiments.

    The same inputs as SquareWave, except that the smoothed staircase is
    integrated on a continuous time grid, so the width of the tanh gates
    (`smoothing`, as a fraction of the pulse width) replaces the number of points
    per pulse (`sampling_factor`).
    """

    def __init__(self, **kwargs):
        # Set default experiment type
        kwargs.setdefault("experiment_type", "SWVtanh")
        super().__init__(**kwargs)

    input_params=ExclusiveDictOption(
        "input_params",
        value_type=numbers.Number,
        target=[
                "omega",
                "Estart",
                "Estop",
                "Eamp",
                "Estep",
                "smoothing"
            ]+["area", "Temp",  "Surface_coverage"],
        doc="Necessary input params for SWVtanh"
    )
class GenericOptions(BaseExperimentOptions):
    """Options specific to Generic experiments."""
    
    def __init__(self, **kwargs):
        # Set default experiment type
        kwargs.setdefault("experiment_type", "Generic")
        super().__init__(**kwargs)
    
    potential_input=TypedOption(
        "potential_input",
        allowed_types=[Basic],
        doc="Potential input constructed using sympy"
    )

    input_params=DictOption(
        "input_params",
        required_keys=["area", "Temp", "Surface_coverage"],
        doc="Necessary input params for generic input"
    )
class SingleExperimentOptions(BaseExperimentOptions):
    _experiment_classes = {
        "FTACV": FTACVOptions,
        "PSV": PSVOptions,
        "DCV": DCVOptions,
        "SquareWave": SquareWaveOptions,
        "Generic":GenericOptions,
        "SWVtanh":SquareWavetanhOptions
    }
    
    def __init__(self, options_handler=None, **kwargs):
        # Extract experiment_type first
        experiment_type = kwargs.get("experiment_type")
        if not experiment_type:
            raise ValueError("experiment_type must be specified")
            
        if experiment_type not in self._experiment_classes:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
        
    
        
        # Create the experiment-specific options object
        base_cls = self._experiment_classes[experiment_type]
        if options_handler is None or options_handler is base_cls:
            # No custom handler: use base directly
            options_cls=base_cls
           
        else:
            # Dynamically create a new class that inherits from both
            # Custom comes first so it overrides base where needed
            CombinedOptions = OptionsMeta(
                f"Combined{options_handler.__name__}{base_cls.__name__}",
                (options_handler, base_cls),
                {}
            )
            options_cls=CombinedOptions
        self._experiment_options = options_cls(**kwargs)
           

        
        # Copy values from experiment options into self
        for name in self._experiment_options.get_option_names():
            if hasattr(self._experiment_options, name):
                setattr(self, name, getattr(self._experiment_options, name))
    
    def as_dict(self):
        """Return a merged dictionary of all options (self + experiment options)."""
        combined = self._experiment_options.as_dict()
        combined.update(super().as_dict())
        return combined
    
    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Dynamically attach properties from all possible experiment types."""
        super().__init_subclass__(**kwargs)
        descriptors_added = set()

        # Add all descriptors from all experiment classes
        for exp_cls in cls._experiment_classes.values():
            for name in exp_cls.get_option_names():
                if name not in descriptors_added:
                    descriptor = getattr(exp_cls, name, None)
                    if descriptor and isinstance(descriptor, OptionDescriptor):
                        setattr(cls, name, descriptor)
                        descriptors_added.add(name)