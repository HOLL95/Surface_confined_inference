import Surface_confined_inference as sci
from Surface_confined_inference.plot import plot_harmonics
from Surface_confined_inference.infer import get_input_parameters
import sys
import matplotlib.pyplot as plt
import numpy as np
ftv = sci.SingleExperiment(
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
ftv.boundaries = {"k0": [1e-3, 200], "E0": [-0.1, 0.06]}
ftv.fixed_parameters = {
    
    "Cdl": 1e-4,
    "gamma": 1e-10,
    "alpha": 0.5,
    "Ru": 100,
}
ftv.dispersion_bins=[2]
ftv.GH_quadrature=True
ftv.optim_list = ["E0","k0"]
nondim_t = ftv.calculate_times(sampling_factor=200, dimensional=False)
dim_t = ftv.dim_t(nondim_t)
dec_amount=8
voltage=ftv.get_voltage(dim_t, dimensional=True)

current = ftv.dim_i(ftv.simulate([0.03,100],nondim_t, ))

noisy_current=sci._utils.add_noise(current, 0.05*max(current))
ftv.Fourier_harmonics=list(range(3, 10))
ftv.Fourier_window="hanning"
results=ftv.Current_optimisation(dim_t, noisy_current,parallel=False,Fourier_filter=True)

print(results)
