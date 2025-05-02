
from ._OptionsDescriptor import (
    BoolOption, EnumOption, NumberOption, SequenceOption, StringOption
)
from ._SingleExperimentOptions import BaseExperimentOptions
from ._OptionsDescriptor import (
    BoolOption, EnumOption, NumberOption, SequenceOption, StringOption
)
import os
class ParallelOptions(BaseExperimentOptions):
    
    num_cpu = NumberOption(
        "num_cpu",
        default=len(os.sched_getaffinity(0)),
        doc="Number of CPUs for parallel simulations for dispersion"
    )
