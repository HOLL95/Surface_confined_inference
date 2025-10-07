import collections.abc
import numbers
from typing import List, Optional, Sequence
from ._OptionsDescriptor import (
    BoolOption, EnumOption, NumberOption, SequenceOption, StringOption, OptionDescriptor, DictOption, ComposedOption, FileOption, DirectoryOption
)
from ._OptionsMeta import OptionsManager, OptionsMeta
class MultiExperimentOptions(OptionsManager):
    """
    Configuration options for multi-experiment electrochemical parameter estimation.

    This class manages options for experiments involving multiple datasets,
    parameter optimization across multiple files, and synthetic data generation.
    It extends OptionsManager to provide validation and management of complex
    multi-experiment configurations.

    """
    SWV_e0_shift=BoolOption("SWV_e0_shift", 
                            default=False, 
                            doc="SWV anodic/cathodic peaks can be at different potentials, this option allows to fit that shift explicitly.")
    file_list=ComposedOption("file_list", 
                            validators=[FileOption],
                            default=None,
                            doc="If fitting to data, list of absolute string paths to the data files")
    boundaries=DictOption("boundaries", 
                        value_type=list,
                        doc="Boundaries for all parameters, required if optimising or if using normalised values"
                        )
    common=DictOption("common",
                    value_type=numbers.Number, 
                    doc="Common input parameters for all individual simulation classes (e.g. Temperature)")
    normalise=BoolOption("normalise", 
                        default=False,
                        doc="If True, evaluation will transform values between 0 and 1 using the values provided in `boundaries`, if False will use the provided value")
    group_list=SequenceOption("group_list", 
                            item_type=dict,
                            default=None,
                            doc="Lists of dicts defining how to group the experiments")
    seperated_parameters=DictOption("seperated_parameters", 
                                    value_type=list,
                                    doc="Parameters that will be different for different experiment simulation groups (for example two different surface coverage values)")
    synthetic=BoolOption("synthetic",
                        default=False,
                        doc="Mode for synthetic studies")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        