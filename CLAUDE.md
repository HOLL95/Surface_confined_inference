# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building
```bash
python -m pip install .
```
This builds the C++ extension via CMake and installs the Python package. The build process:
1. Compiles C++ ODE solver using SUNDIALS/CVODE
2. Creates Python bindings via pybind11
3. Installs the SurfaceODESolver C++ module

### Testing
```bash
pytest --cov=Surface_confined_inference --cov-config=.coveragerc --cov-report=xml tests/
```
Run the test suite with coverage reporting. The main test file is `tests/SingleExperiment_test.py`.

### Verification
```bash
python example_usage.py
```
Runs a comprehensive integration test that verifies the installation works correctly.

### Linting and Type Checking
```bash
ruff check Surface_confined_inference/
```
Code linting is configured in `pyproject.toml` with ruff.

## Architecture Overview

### Hybrid Python/C++ Design
This project combines Python for high-level interfaces with C++ for computationally intensive ODE solving:

- **Python Layer**: Experiment setup, parameter management, inference, plotting
- **C++ Layer**: Fast ODE simulation using SUNDIALS/CVODE library
- **Communication**: pybind11 provides seamless Python-C++ integration

### Core Directory Structure
- `Surface_confined_inference/_core/` - Main simulation classes and experiment types
- `Surface_confined_inference/_core/_Options/` - Configuration management system using mixins
- `Surface_confined_inference/_core/_Handlers/` - Parameter and voltammetry handling
- `Surface_confined_inference/infer/` - Bayesian inference utilities
- `Surface_confined_inference/plot/` - Specialized plotting for electrochemistry
- `C_src/` - C++ ODE solver implementation and CMake build system

### Key Classes Architecture
1. **BaseExperiment**: Foundation class with serialization and core functionality
2. **SingleExperiment**: Individual electrochemical experiments (FTACV, DCV, etc.)
3. **MultiExperiment**: Multiple experiment handling with parallel execution
4. **AxInterface**: Integration with Meta's Ax platform for optimization
5. **Options Classes**: Mixin-based configuration system using descriptors

### Experiment Types
The system supports multiple electrochemical techniques:
- **FTACV**: Fourier Transform Alternating Current Voltammetry
- **DCV**: Direct Current Voltammetry
- **SWV**: Square Wave Voltammetry
- **PSV**: Purely Sinusoidal Voltammetry

## Critical Dependencies

### CVODE Requirement
The C++ ODE solver requires SUNDIALS/CVODE to be manually installed:
1. Download SUNDIALS from LLNL website
2. Build with CMake in a `builddir`
3. Set environment variable: `export CVODE_PATH="/path/to/builddir"`
4. Alternative: Edit `C_src/CMakeLists.txt` directly

### HPC Cluster Support
Built-in configurations for specific clusters:
- **VIKING cluster**: Set `IN_VIKING=true` environment variable
- **ARC cluster**: Set `IN_ARC=true` environment variable
These automatically configure SUNDIALS and pybind11 paths.

## Development Patterns

### Options System
The codebase uses a sophisticated mixin-based options system:
- `OptionsAwareMixin`: Provides options functionality to classes
- `OptionsMeta`: Metaclass for automatic option descriptor creation
- `OptionsDescriptor`: Individual option validation and storage
- Options are automatically validated and provide helpful error messages

### Nondimensionalization System
All simulations use nondimensional variables internally:
- `NDParams` class handles dimensional analysis
- `dim_t()` and `dim_i()` methods convert between dimensional/nondimensional
- This improves numerical stability and solver performance

### Parameter Management
- `optim_list`: Parameters to optimize during fitting
- `fixed_parameters`: Parameters held constant
- `boundaries`: Min/max bounds for optimization parameters
- Parameters are automatically validated against the experiment type

### Simulation Workflow
1. Create experiment with parameters dictionary
2. Set optimization parameters and boundaries 
3. Calculate time points with `calculate_times()` if synthetic, else experimental times and currents are provided by the user
4. Run simulation with `simulate(parameters, times)`
5. Convert results to dimensional units if needed

### Dispersion Modeling
The system supports parameter dispersion modeling:
- `Dispersion` class handles parameter distributions
- `dispersion_bins` controls distribution discretization
- `GH_quadrature` enables Gauss-Hermite quadrature integration

## Testing Notes
- Main test suite focuses on `SingleExperiment` functionality
- Tests use numpy arrays stored in `tests/testdata/`
- Coverage reporting is configured for the main package
- GitHub Actions runs tests on Python 3.8-3.12

## Common Issues
- **CVODE not found**: Ensure CVODE_PATH is set correctly
- **Build failures**: Check CMake and C++ compiler availability
- **Import errors**: Verify the C++ extension compiled successfully
- **Simulation crashes**: Check parameter bounds and numerical stability