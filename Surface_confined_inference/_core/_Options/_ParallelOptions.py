
from ._OptionsDescriptor import (
    BoolOption, EnumOption, NumberOption, SequenceOption, StringOption
)
from ._SingleExperimentOptions import SingleExperimentOptions
from ._OptionsDescriptor import (
    BoolOption, EnumOption, NumberOption, SequenceOption, StringOption
)
class ParallelOptions(SingleExperimentOptions):
    num_cpu = NumberOption(
        "num_cpu",
        default=1,
        doc="Number of CPUs for parallel simulations for dispersion"
    )
