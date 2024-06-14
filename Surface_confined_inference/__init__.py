import numpy as np
from dataclasses import dataclass
from ._core._Experiments import SingleExperiment
from ._core._InputChecking import check_input_dict
from ._core._Nondimensionalise import NDParams
from ._core._Dispersion import Dispersion