import Surface_confined_inference as sci
import sys
import matplotlib.pyplot as plt
import numpy as np

# Define experimental parameters for FTACV (Fourier Transform Alternating Current Voltammetry)
inputs={
        "E_reverse": 0.3,        # Reverse potential (V)
        "omega": 10,             # AC frequency (Hz)
        "delta_E": 0.15,         # AC amplitude (V)
        "area": 0.07,            # Electrode area (cm^2)
        "Temp": 298,             # Temperature (K)
        "N_elec": 1,             # Number of electrons transferred
        "phase": 0,              # Phase offset (radians)
        "Surface_coverage": 1e-10,  # Initial surface coverage (mol/cm^2)
        "v": 25e-3,              # Scan rate (V/s)
    }
# Create a SingleExperiment object for FTACV with parameter dispersion modeling
ftv = sci.SingleExperiment(
    "FTACV",              # Experiment type
    inputs,               # Dictionary of experimental parameters
    parallel_cpu=8,       # Number of CPU cores for parallel processing
    problem="inverse"     # Inverse problem setup for parameter inference
)

# Set optimization boundaries for parameters to be fitted
ftv.boundaries = {"k0": [1e-3, 200], "E0_std": [1e-3, 0.06]}

# Define fixed parameters that won't be optimized
ftv.fixed_parameters = {
    "E0_mean":0.1,   # Mean formal potential (V)
    "Cdl": 1e-4,     # Double layer capacitance (F/cm^2)
    "gamma": 1e-10,  # Surface coverage (mol/cm^2)
    "alpha": 0.5,    # Charge transfer coefficient
    "Ru": 100,       # Uncompensated resistance (Ohms)
}

# Configure dispersion modeling
ftv.dispersion_bins=[15]      # Number of bins for discretizing the parameter distribution
ftv.GH_quadrature=True        # Use Gauss-Hermite quadrature for integration
ftv.optim_list = ["E0_std","k0"]  # Parameters to optimize: E0 standard deviation and electron transfer rate
# Calculate time points for simulation
nondim_t = ftv.calculate_times(sampling_factor=200, dimensional=False)  # Nondimensional time array
dim_t = ftv.dim_t(nondim_t)  # Convert to dimensional time (seconds)

dec_amount=8  # Decimation amount (unused in this example)

# Get the applied voltage waveform
voltage=ftv.get_voltage(dim_t)

# Get the DC component of voltage (AC amplitude set to 0)
dc_voltage=ftv.get_voltage(dim_t,
                        input_parameters={key:inputs[key]if key!="delta_E" else 0 for key in inputs.keys()},
                        )

# Simulate current with dispersion: E0_std=0.05 V, k0=100 s^-1
current = ftv.dim_i(ftv.simulate([0.05,100],nondim_t, ))

# Create a non-dispersed experiment for comparison
non_disped_ftv=sci.SingleExperiment(
    "FTACV",
    inputs,
    problem="forwards"  # Forward problem setup (no inference)
)

# Specify all electrochemical parameters to simulate
non_disped_ftv.optim_list=["E0","k0","gamma","Cdl","alpha","Ru"]

# Simulate current without dispersion: E0=0.1 V, k0=100 s^-1, gamma=1e-10 mol/cm^2,
# Cdl=1e-4 F/cm^2, alpha=0.5, Ru=100 Ohms
non_disped_current=non_disped_ftv.dim_i(non_disped_ftv.simulate([0.1,100, 1e-10, 1e-4, 0.5, 100 ],nondim_t, ))

# Test serialization: save the experiment to JSON
ftv.save_class("json_test")

# Test deserialization: load the experiment from JSON
load=sci.BaseExperiment.from_json("json_test.json")

# Plot comparison of dispersed vs non-dispersed simulations
plt.plot(dc_voltage, non_disped_current, label="No dispersion")
plt.plot(dc_voltage,current, label="Original class")

# Verify that the loaded class produces identical results
plt.plot(dc_voltage,load.dim_i(load.simulate([0.05, 100], nondim_t)), color="black", linestyle="--", label="Loaded class")
plt.legend()
plt.show()
