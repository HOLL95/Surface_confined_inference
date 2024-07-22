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
ftv.boundaries = {"k0": [1e-3, 200], 
                    "E0": [-0.1, 0.06],
                    "Cdl": [1e-5, 1e-3],
                    "gamma": [1e-11, 1e-9],
                    "Ru": [1, 1e3],}
ftv.fixed_parameters = {
    "alpha":0.5,
}

ftv.optim_list = ["E0","k0", "Cdl", "gamma",  "Ru"]

nondim_t = ftv.calculate_times(sampling_factor=200, dimensional=False)
dim_t = ftv.dim_t(nondim_t)
voltage=ftv.get_voltage(dim_t, dimensional=True)
current = ftv.dim_i(ftv.simulate([0.03,100, 1e-4, 1e-10, 100],nondim_t, ))

noisy_current=sci._utils.add_noise(current, 0.05*max(current))
with open("test_inference.txt", "w") as f:
    np.savetxt(f, np.column_stack((dim_t, noisy_current, voltage)))
results=ftv.Current_optimisation(dim_t, noisy_current,
                                parallel=True,
                                Fourier_filter=False, 
                                runs=6, 
                                save_to_directory="Results", 
                                threshold=1e3, 
                                unchanged_iterations=1,
                                save_csv=True)


