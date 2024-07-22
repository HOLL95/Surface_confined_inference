import Surface_confined_inference as sci
from Surface_confined_inference.plot import plot_harmonics
from Surface_confined_inference.infer import get_input_parameters
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
slurm_class = sci.SingleSlurmSetup(
    "FTACV",
    {
        "E_start": -0.4,
        "E_reverse": 0.3,
        "omega": 10,
        "delta_E": 0.15,
        "area": 0.07,
        "Temp": 298,
        "N_elec": 1,
        "phase": 0,
        "Surface_coverage": 1e-10,
        "v": 25e-3,
    },
)
slurm_class.boundaries = {"k0": [1e-3, 200], 
                    "E0": [-0.1, 0.06],
                    "Cdl": [1e-5, 1e-3],
                    "gamma": [1e-11, 1e-9],
                    "Ru": [1, 1e3],}
slurm_class.fixed_parameters = {
    "alpha":0.5,
}

slurm_class.optim_list = ["E0","k0", "Cdl", "gamma",  "Ru"]
slurm_class.setup(
    datafile="test_inference.txt",
    cpu_ram="8G",
    time="0-00:10:00",
    runs=5, 
    threshold=1e3, 
    unchanged_iterations=1   
)