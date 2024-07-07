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
ftv.boundaries = {"k0": [1e-3, 200], "E0_std": [1e-3, 0.06]}
ftv.fixed_parameters = {
    "E0_mean":0.1,
    "Cdl": 1e-4,
    "gamma": 1e-10,
    "alpha": 0.5,
    "Ru": 100,
}
ftv.dispersion_bins=[2]
ftv.GH_quadrature=True
ftv.optim_list = ["E0_std","k0"]
nondim_t = ftv.calculate_times(sampling_factor=200, dimensional=False)
dim_t = ftv.dim_t(nondim_t)
dec_amount=8
voltage=ftv.get_voltage(dim_t, dimensional=True)

current = ftv.dim_i(ftv.simulate(nondim_t, [0.03,100]))
sci.infer.get_input_parameters(dim_t, sci._utils.add_noise(voltage, 0.01*max(voltage)), current, "FTACV", plot_results=True)
current_fig, current_ax = plt.subplots()
current_ax.plot(dim_t, current)
current_ax.set_xlabel("Time (s)")
current_ax.set_ylabel("Current (A)")
plot_harmonics(
    Synthetic_data={"time": dim_t, "current": current},
    hanning=False,
    plot_func=abs,
    harmonics=list(range(3, 9)),
    xlabel="Time (s)",
    ylabel="Current (A)"
)
plt.show()
