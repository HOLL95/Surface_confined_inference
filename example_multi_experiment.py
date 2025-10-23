"""
MultiExperiment Example Script

This script demonstrates how to use the MultiExperiment class to set up and simulate
multiple electrochemical experiments simultaneously. It combines both FTACV (Fourier
Transform Alternating Current Voltammetry) and SWV (Square Wave Voltammetry) experiments
with different parameters, showing how to:
1. Configure multiple experiments with shared and unique parameters
2. Set up Fourier transform analysis for FTACV experiments
3. Define optimization parameters and boundaries
4. Group experiments for analysis and visualization
5. Run simulations across all experiments
"""

import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import os

# Dictionary to hold all experiment configurations
experiments_dict = {}

# =============================================================================
# FTACV Experiment Setup
# =============================================================================
# Define FTACV experiments at three different frequencies (3, 9, and 15 Hz)
# All experiments share the same voltage parameters except for frequency (omega)
dictionary_list = [
    # 3 Hz FTACV experiment
    {'E_start': np.float64(-0.89), 'E_reverse': np.float64(0), 'omega': np.float64(3.0), 'phase': np.float64(0), 'delta_E': np.float64(0.28), 'v': np.float64(0.059)},
    # 9 Hz FTACV experiment
    {'E_start': np.float64(-0.89), 'E_reverse': np.float64(0), 'omega': np.float64(9.0), 'phase': np.float64(0), 'delta_E': np.float64(0.28), 'v': np.float64(0.059)},
    # 15 Hz FTACV experiment
    {'E_start': np.float64(-0.89), 'E_reverse': np.float64(0), 'omega': np.float64(15.0), 'phase': np.float64(0), 'delta_E': np.float64(0.28), 'v': np.float64(0.059)},
]
# Fourier transform options for FTACV experiments
# These control how the time-series data is transformed into frequency domain
FT_options = dict(
    Fourier_fitting=True,              # Enable Fourier transform analysis
    Fourier_window="hanning",          # Window function to reduce spectral leakage
    top_hat_width=0.25,                # Width of the top-hat filter in frequency domain
    Fourier_function="abs",            # Use absolute value of Fourier transform
    Fourier_harmonics=list(range(3, 10)),  # Fit harmonics 3 through 9
    dispersion_bins=[30],              # Number of bins for parameter dispersion modeling
    optim_list=["E0", "k0", "gamma", "Ru", "Cdl", "alpha"]  # Parameters to optimize
)

# Parameter boundaries for optimization
# These define the search space for each electrochemical parameter
boundaries = {
    "E0": [-0.6, -0.3],          # Formal potential (V)
    "k0": [1e-3, 1000],          # Standard rate constant (s^-1)
    "gamma": [1e-11, 1e-10],     # Surface coverage (mol/cm^2)
    "Ru": [1e-3, 1000],          # Uncompensated resistance (Ohms)
    "Cdl": [1e-6, 5e-4],         # Double layer capacitance (F)
    "alpha": [0.4, 0.6]          # Charge transfer coefficient (dimensionless)
}
# Labels for the three FTACV experiments
labels = ["3_Hz", "9_Hz", "15_Hz"]

# Common parameters shared across ALL experiments (both FTACV and SWV)
# These are fixed experimental conditions
common = {
    "Temp": 278,              # Temperature (K)
    "N_elec": 1,              # Number of electrons transferred in redox reaction
    "area": 0.036,            # Electrode area (cm^2)
    "Surface_coverage": 1e-10 # Surface coverage (mol/cm^2)
}

# Worst case simulation to calculate hypervolume threshold for multi experiment optimisation
# Format: [E0, k0, gamma, Ru, Cdl, alpha]
zero_ft = [-0.425, 100, 8e-11, 100, 1.8e-4, 0.5]

# Construct the FTACV experiments and add them to the experiments dictionary. These can be anything, and will be used as
# identifiers when grouping experiments below
for i in range(0, len(labels)):
    experiments_dict = sci.construct_experimental_dictionary(
        experiments_dict,
        {**{"Parameters": dictionary_list[i]}, **{"Options": FT_options}, "Zero_params": zero_ft},
        "FTACV",     # Experiment type
        labels[i],   # Frequency label (e.g., "3_Hz")
        "280_mV"     # Voltage amplitude label
    )

# =============================================================================
# Square Wave Voltammetry (SWV) Experiment Setup
# =============================================================================
# Define 14 different frequencies for SWV experiments (in Hz)
sw_freqs = [65, 75, 85, 100, 115, 125, 135, 145, 150, 175, 200, 300, 400, 500]

# Defining threshold parameters - this involves subtracting the Fardaic peak, and smoothing 
# the resulting data
zero_sw = {
    "potential_window": [-0.425-0.15, -0.425+0.15],  # Window around E0 for baseline
    "thinning": 10,       # Data point reduction factor
    "smoothing": 20       # Smoothing window size
}

# SWV experiments will be run in both anodic (oxidation) and cathodic (reduction) directions
directions = ["anodic", "cathodic"]

# Configuration for each scan direction
directions_dict = {
    "anodic": {"v": 1, "E_start": -0.8},    # Scan from negative to positive potential
    "cathodic": {"v": -1, "E_start": 0}     # Scan from positive to negative potential
}

# SWV-specific options
sw_options = dict(
    square_wave_return="net",  # Return net current (backward - forward)
    dispersion_bins=[30],      # Number of bins for parameter dispersion modeling
    optim_list=["E0", "k0", "gamma", "alpha"]  # Parameters to optimize (fewer than FTACV as can't model contribution of Cdl or Ru)
)

# Construct all SWV experiments
for i in range(0, len(sw_freqs)):
    for j in range(0, len(directions)):
        params = {
            "omega": sw_freqs[i],            # Frequency (Hz)
            "scan_increment": 5e-3,          # Potential step size (V)
            "delta_E": 0.8,                  # Total potential scan range (V)
            "SW_amplitude": 5e-3,            # Square wave pulse amplitude (V)
            "sampling_factor": 200,          # Points per square wave cycle
            "E_start": directions_dict[directions[j]]["E_start"],  # Starting potential
            "v": directions_dict[directions[j]]["v"]               # Scan direction (+1 or -1)
        }
        experiments_dict = sci.construct_experimental_dictionary(
            experiments_dict,
            {**{"Parameters": params}, **{"Options": sw_options}, "Zero_params": zero_sw},
            "SWV",                           # Experiment type
            "{0}_Hz".format(sw_freqs[i]),    # Frequency label
            directions[j]                    # Direction label ("anodic" or "cathodic")
        )



# =============================================================================
# Experiment Grouping for Visualization
# =============================================================================
# Define how experiments should be grouped when fitting. Multiobjetive optimisation generally
# can only optimise over fewer objectives than we have experiments, so we scalararise (and scale)
# the experiments
# Each group specification selects experiments from the total list
group_list = [
    # Group 1: FTACV time-series for frequencies < 10 Hz (3 and 9 Hz)
    {
        "experiment": "FTACV",
        "type": "ts",  # Time-series data
        "numeric": {"Hz": {"lesser": 10}, "mV": {"equals": 280}},
        "scaling": {"divide": ["omega", "delta_E"]}  # Normalize by frequency and amplitude
    },
    # Group 2: FTACV Fourier transform for frequencies < 10 Hz
    {
        "experiment": "FTACV",
        "type": "ft",  # Fourier transform data
        "numeric": {"Hz": {"lesser": 10}, "mV": {"equals": 280}},
        "scaling": {"divide": ["omega", "delta_E"]}
    },
    # Group 3: FTACV time-series for 15 Hz only
    {
        "experiment": "FTACV",
        "type": "ts",
        "numeric": {"Hz": {"equals": 15}, "mV": {"equals": 280}},
        "scaling": {"divide": ["omega", "delta_E"]}
    },
    # Group 4: FTACV Fourier transform for 15 Hz only
    {
        "experiment": "FTACV",
        "type": "ft",
        "numeric": {"Hz": {"equals": 15}, "mV": {"equals": 280}},
        "scaling": {"divide": ["omega", "delta_E"]}
    },
    # Group 5: All anodic SWV experiments
    {
        "experiment": "SWV",
        "match": ["anodic"],
        "type": "ts",
        "scaling": {"divide": ["omega"]}  # Normalize by frequency only
    },
    # Group 6: All cathodic SWV experiments
    {
        "experiment": "SWV",
        "match": ["cathodic"],
        "type": "ts",
        "scaling": {"divide": ["omega"]}
    },
]

# =============================================================================
# MultiExperiment Initialization and Simulation
# =============================================================================
# Create MultiExperiment object with all configured experiments
# Total: 3 FTACV experiments + 28 SWV experiments = 31 experiments
cls = sci.MultiExperiment(
    experiments_dict,
    common=common,          # Shared parameters across all experiments
    synthetic=True,         # Flag to indicate data is synthetic
    normalise=True,         # When simulating, parameters are normalised between 0 and 1 using boundaries
    boundaries=boundaries   # Parameter search boundaries
)

# Get list of all parameters that will be optimized
params = cls._all_parameters

# Assign the grouping configuration 
cls.group_list = group_list

# Load experimental data files from test directory
#The (.txt) files need to be:
#1) In dimensional form
#2) Time in column 1, Current in column 2
#3) Labelled according to the group structure defined above 
# It needs an experimental signifier, numerical values in the format number_unit
# and these signifiers seperated by dashes (-)
#(e.g. FTACV-3_Hz-280_mV.txt)
fileloc = os.path.join(os.getcwd(), "tests/testdata/multi")
cls.file_list = [os.path.join(fileloc, file) for file in os.listdir(fileloc)]

# Check the grouping and scaling operations
cls.check_grouping()

# Generate random parameter values for demonstration (between 0 and 1 as normalisation is on )
# In a real scenario, these would come from optimization or fitting
sim_param_vals = np.random.rand(len(params))

# Run simulations across all 31 experiments with the random parameters
# This demonstrates that the MultiExperiment infrastructure is working
cls.results(parameters=sim_param_vals)


#Setup MultiExperiment inference to run locally
ax_class=sci.AxInterface(name="Example_submission",
			independent_runs=1, #Pareto points are pooled from all independent runs
			num_iterations=100,
			num_cpu=1,#Only for dispersion simulation, won't get any speedup if it's non-dispersed
			simulate_front=False,#Simulate each pareto point for results
			in_cluster=False)
ax_class.setup_client(cls)
#Uncomment this to run inference locally
#ax_class.experiment()
#These processes can be quite hard to stop without a SLURM manager
#See https://github.com/facebookincubator/submitit/issues/1766

