# Surface Confined Inference

A Python package for simulating, analyzing, and performing inference on surface-confined electrochemical systems. This project combines C++ ODE solvers with Python interfaces to enable efficient parameter estimation for various voltammetry techniques for surfaceo-confined systems


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Experiment Types](#experiment-types)
- [Architecture](#architecture)
- [Examples](#examples)
- [Testing](#testing)

## Overview

Surface Confined Inference provides a comprehensive framework for studying electrochemical systems where the redox-active species is confined to the electrode surface (as opposed to freely diffusing in solution). The package is designed for researchers who need to:

- Simulate electrochemical experiments with realistic instrumental effects (capacitance, resistance)
- Fit experimental data to extract kinetic and thermodynamic parameters
- Perform multi-experiment optimisation
- Model parameter distributions (dispersion) in heterogeneous systems


### Electrochemical Techniques

The package supports multiple voltammetric techniques, each with specific advantages:

- **FTACV (Fourier Transform AC Voltammetry)**: Superimposes a sinusoidal perturbation on a DC ramp, allowing extraction of harmonics that are sensitive to different kinetic regimes
- **DCV (Direct Current Voltammetry)**: Traditional cyclic voltammetry with linear potential sweep
- **SWV (Square Wave Voltammetry)**: Applies a square wave on a staircase ramp
- **PSV (Purely Sinusoidal Voltammetry)**: Pure sinusoidal modulation for fundamental harmonic analysis


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
make install  
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
python -m pip install -e . 
```

#### Step 4: Verify installation

```bash
python example_usage.py
```


If successful, you should see a matplotlib plot comparing dispersed and non-dispersed simulations.

### HPC Cluster Support

For specific HPC clusters, all installation can be achieved by running these commands

```bash
# VIKING cluster (University of York)
source HPC_installers/Viking/viking_setup.sh

# ARC cluster (University of Oxford)
source HPC_installers/ARC/arc_setup.sh

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


