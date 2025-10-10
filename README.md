# Surface Confined Inference

A Python package for simulating, analyzing, and performing inference on surface-confined electrochemical systems. This project combines C++ ODE solvers with Python interfaces to enable efficient parameter estimation for various voltammetry techniques for surfaceo-confined systems

[![Tests](https://github.com/HOLL95/Surface_confined_inference/actions/workflows/test.yml/badge.svg)](https://github.com/HOLL95/Surface_confined_inference/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiment Types](#experiment-types)
- [Architecture](#architecture)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

Surface Confined Inference provides a comprehensive framework for studying electrochemical systems where the redox-active species is confined to the electrode surface (as opposed to freely diffusing in solution). The package is designed for researchers who need to:

- Simulate electrochemical experiments with realistic instrumental effects (capacitance, resistance)
- Fit experimental data to extract kinetic and thermodynamic parameters
- Perform Bayesian parameter inference with uncertainty quantification
- Handle multiple experiments simultaneously with shared parameters
- Model parameter distributions (dispersion) in heterogeneous systems


### Electrochemical Techniques

The package supports multiple voltammetric techniques, each with specific advantages:

- **FTACV (Fourier Transform AC Voltammetry)**: Superimposes a sinusoidal perturbation on a DC ramp, allowing extraction of harmonics that are sensitive to different kinetic regimes
- **DCV (Direct Current Voltammetry)**: Traditional cyclic voltammetry with linear potential sweep
- **SWV (Square Wave Voltammetry)**: Applies a square wave on a staircase ramp
- **PSV (Purely Sinusoidal Voltammetry)**: Pure sinusoidal modulation for fundamental harmonic analysis

### Mathematical Framework

The package uses a **nondimensionalization scheme** to improve numerical stability and solver performance. All simulations internally use dimensionless variables, with automatic conversion to/from physical units. This is handled transparently by the `NDParams` class.

Key parameters include:
- **E0**: Formal potential (V)
- **k0**: Standard electron transfer rate constant (s⁻¹)
- **α**: Charge transfer coefficient (dimensionless, typically ~0.5)
- **Γ**: Surface coverage (mol/cm²)
- **Ru**: Uncompensated resistance (Ω)
- **Cdl**: Double-layer capacitance (F/cm²)

## Installation

### Prerequisites

- Python 3.10 or higher
- CMake 3.12 or higher
- C++ compiler with C++11 support
- Git

### Required: SUNDIALS/CVODE Installation

This package requires the SUNDIALS library (specifically CVODE) for ODE solving. You must install it manually before building the Python package.

#### Step 1: Download SUNDIALS

Download SUNDIALS from the [Lawrence Livermore National Laboratory website](https://computing.llnl.gov/projects/sundials/sundials-software).

#### Step 2: Build SUNDIALS

```bash
# Extract the archive
tar -xzf sundials-x.y.z.tar.gz
cd sundials-x.y.z

# Create build directory
mkdir builddir
cd builddir

# Configure and build
cmake ..
make
make install  # Optional, may require sudo
```

#### Step 3: Set CVODE Path

Tell the build system where to find CVODE using one of these methods:

**Option A: Environment variable (recommended)**
```bash
export CVODE_PATH="/absolute/path/to/sundials/builddir"
```

Add this to your `~/.bashrc` or `~/.zshrc` to make it permanent:
```bash
echo 'export CVODE_PATH="/absolute/path/to/sundials/builddir"' >> ~/.bashrc
source ~/.bashrc
```

**Option B: Direct CMake configuration**

Edit `Surface_confined_inference/C_src/CMakeLists.txt` and modify:
```cmake
set (SUNDIALS_DIR /absolute/path/to/sundials/builddir)
```

**Verify the path is set:**
```bash
echo $CVODE_PATH
# Should output: /absolute/path/to/sundials/builddir
```




### Installing Surface Confined Inference

#### Step 1: Clone the repository

```bash
git clone https://github.com/HOLL95/Surface_confined_inference.git --recurse-submodules
cd Surface_confined_inference
```

**Note**: The `--recurse-submodules` flag is important as it includes the pybind11 submodule.

#### Step 2: Create a virtual environment (recommended)

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

#### Step 3: Install the package

```bash
python -m pip install . 
```

This will:
1. Compile the C++ ODE solver using CMake
2. Create Python bindings via pybind11
3. Install the Python package and dependencies

#### Step 4: Verify installation

```bash
python example_usage.py
```


If successful, you should see a matplotlib plot comparing dispersed and non-dispersed simulations.

### HPC Cluster Support

For specific HPC clusters, set these environment variables to use pre-configured paths:

```bash
# VIKING cluster (University of York)
source HPC_installers/Viking/viking_setup.sh

# ARC cluster (University of Oxford)
source HPC_installers/ARC/arc_setup.sh

```

## Quick Start

### Basic Single Experiment

Here's a minimal example to simulate an FTACV experiment:

```python
import Surface_confined_inference as sci
import matplotlib.pyplot as plt

# Define experimental parameters
params = {
    "E_reverse": 0.3,      # Reverse potential (V)
    "omega": 10,           # AC frequency (Hz)
    "delta_E": 0.15,       # AC amplitude (V)
    "area": 0.07,          # Electrode area (cm²)
    "Temp": 298,           # Temperature (K)
    "N_elec": 1,           # Number of electrons
    "v": 25e-3,            # Scan rate (V/s)
}

# Create experiment
exp = sci.SingleExperiment("FTACV", params)



exp.fixed_parameters = {
    "Cdl": 1e-4,
    "gamma": 1e-10,
    "alpha": 0.5,
    "Ru": 100,
}

exp.optim_list = ["E0", "k0"]

# Generate time points and simulate
times = exp.calculate_times(sampling_factor=200, dimensional=False)
current = exp.simulate([0.1, 100], times)  # E0=0.1 V, k0=100 s⁻¹

# Convert to dimensional units and plot
time_s = exp.dim_t(times)
current_A = exp.dim_i(current)
voltage_V = exp.get_voltage(time_s)

plt.plot(voltage_V, current_A)
plt.xlabel('Potential (V)')
plt.ylabel('Current (A)')
plt.show()
```



### Saving and Loading

Experiments and the associated configuration options can be serialized to JSON:

```python
# Save experiment configuration
exp.save_class("my_experiment")

# Load in another session
loaded_exp = sci.BaseExperiment.from_json("my_experiment.json")

# Verify it works
current_loaded = loaded_exp.simulate([0.05, 100], times)
```

## Experiment Types

### FTACV (Fourier Transform AC Voltammetry)


**Key parameters**:
- `omega`: AC frequency (Hz)
- `delta_E`: AC amplitude (V)
- `v`: DC scan rate (V/s)
- `E_reverse`: Reverse potential (V)
- `phase` : Sinusoidal phase (rads)

**Configuration**:
```python
exp = sci.SingleExperiment("FTACV", params)
exp.Fourier_fitting = True
exp.Fourier_harmonics = list(range(3, 10))  # Analyze harmonics 3-9
exp.Fourier_window = "hanning"              # Reduce spectral leakage
```

### DCV (Direct Current Voltammetry)


**Key parameters**:
- `v`: Scan rate (V/s)
- `E_reverse`: Reverse potential (V)
- `E_start`: Starting potential (V)

**Configuration**:
```python
exp = sci.SingleExperiment("DCV", params)
```

### SWV (Square Wave Voltammetry)



**Key parameters**:
- `omega`: Square wave frequency (Hz)
- `SW_amplitude`: Pulse amplitude (V)
- `scan_increment`: Staircase step (V)
- `delta_E` : Voltage range (V)
- `E_start` : Starting potential (V)
- `scan_direction` : Increasing or decreasing from E_start

**Configuration**:
```python
exp = sci.SingleExperiment("SWV", params)
exp.square_wave_return = "net"  # or "forward", "backward"
```

### PSV (Purely Sinusoidal Voltammetry)



**Key parameters**:
- `omega`: AC frequency (Hz)
- `delta_E`: AC amplitude (V)
- `phase` : Sinusoidal phase (rads)

**Configuration**:
```python
exp = sci.SingleExperiment("PSV", params)
```

## Architecture

The package uses a hybrid Python/C++ architecture:

```
┌─────────────────────────────────────────┐
│         Python Layer (User API)         │
│                                         │
│  • Experiment setup & configuration     │
│  • Parameter management                 │
│  • Bayesian inference                   │
│  • Plotting & visualization             │
└─────────────────────────────────────────┘
                   │
                   ▼ pybind11
┌─────────────────────────────────────────┐
│      C++ Layer (High Performance)       │
│                                         │
│  • ODE solving (SUNDIALS/CVODE)        │
│  • Numerical integration                │
└─────────────────────────────────────────┘
```

### Key Classes

- **`SingleExperiment`**: Main interface for individual experiments
- **`MultiExperiment`**: Handle multiple experiments with shared parameters
- **`BaseExperiment`**: Foundation with serialization support
- **`ParameterHandler`**: Manages optimization parameters and boundaries
- **`NDParams`**: Handles dimensional analysis and nondimensionalization
- **`Dispersion`**: Models parameter distributions

### Options System



```python
exp = sci.SingleExperiment("FTACV", params,
    parallel_cpu=8,           # Number of CPU cores
    problem="inverse",        # "inverse" or "forwards"
    Fourier_fitting=True,
    dispersion_bins=[15]
)

# Options are validated and provide helpful error messages. The can be accessed and set via exp.option
# and will be saved using the `save_class()` method
```

## Examples


### 1. Single Experiment (`example_usage.py`)

Demonstrates:
- Basic experiment setup
- Parameter dispersion modeling
- Gauss-Hermite quadrature integration
- Serialization/deserialization
- Comparison of dispersed vs. non-dispersed simulations

Run with:
```bash
python example_usage.py
```

### 2. Multiple Experiments (`example_multi_experiment.py`)

Demonstrates:
- Combining FTACV and SWV experiments
- Multi-frequency experimental design
- Experiment grouping for analysis
- Scaling and normalization strategies

Run with:
```bash
python example_multi_experiment.py
```

## Testing

The package includes a test suite with integration tests.

### Running Tests

```bash
# Run all tests 
cd tests/
python -m pytest .
```


