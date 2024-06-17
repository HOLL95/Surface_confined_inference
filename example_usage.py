
import Surface_confined_inference as sci
import sys
import matplotlib.pyplot as plt
ftv=sci.SingleExperiment("FTACV", {"E_start":-0.4, "E_reverse":0.3, "omega":9, "delta_E":0.15, "area":0.07, "Temp":298, "N_elec":1, "phase":0, "Surface_coverage":1e-10, "v":25e-3})
ftv.boundaries={"k0":[1e-3, 200], "E0_std":[1e-3, 0.06]}
ftv.fixed_parameters={  
                        "E0_mean":0.1,
                        "Cdl":1e-4,
                        "gamma":1e-10,
                        "alpha":0.5,
                        "Ru":100, }
ftv.dispersion_bins=[16]
ftv.GH_quadrature=True
ftv.optim_list=["E0_std", "k0"]
nondim_t=ftv.calculate_times(sampling_factor=200, dimensional=False)
dim_t=ftv.dim_t(nondim_t)
current=ftv.simulate(nondim_t, [0.05, 10])
plt.plot(dim_t, current)
plt.show()