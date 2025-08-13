import Surface_confined_inference as sci
import sys
import matplotlib.pyplot as plt
import numpy as np
inputs={
        "E_start": -0.4,
        "E_reverse": 0.3,
        "omega": 1,
        "delta_E": 0.15,
        "area": 0.07,
        "Temp": 298,
        "N_elec": 1,
        "phase": 0,
        "Surface_coverage": 1e-10,
        "v": 25e-3,
    }
ftv = sci.SingleExperiment(
    "FTACV",
    inputs,
    parallel_cpu=8,
    problem="inverse"
)
ftv.boundaries = {"k0": [1e-3, 200], "E0_std": [1e-3, 0.06]}
ftv.fixed_parameters = {
    "E0_mean":0.1,
    "Cdl": 1e-4,
    "gamma": 1e-10,
    "alpha": 0.5,
    "Ru": 100,
}
ftv.dispersion_bins=[15]
ftv.GH_quadrature=True
ftv.optim_list = ["E0_std","k0"]
nondim_t = ftv.calculate_times(sampling_factor=200, dimensional=False)
dim_t = ftv.dim_t(nondim_t)
dec_amount=8
voltage=ftv.get_voltage(dim_t)
dc_voltage=ftv.get_voltage(dim_t, 
                        input_parameters={key:inputs[key]if key!="delta_E" else 0 for key in inputs.keys()},
                        )
current = ftv.dim_i(ftv.simulate([0.05,100],nondim_t, ))

non_disped_ftv=sci.SingleExperiment(
    "FTACV",
    inputs,
    problem="forwards"
)
non_disped_ftv.optim_list=["E0","k0","gamma","Cdl","alpha","Ru"]
non_disped_current=non_disped_ftv.dim_i(non_disped_ftv.simulate([0.1,100, 1e-10, 1e-4, 0.5, 100 ],nondim_t, ))

ftv.save_class("json_test")
load=sci.BaseExperiment.from_json("json_test.json")
plt.plot(non_disped_current, label="No dispersion")
plt.plot(current, label="Original class")

plt.plot(load.dim_i(load.simulate([0.05, 100], nondim_t)), color="black", linestyle="--", label="Loaded class")
plt.legend()
plt.show()
