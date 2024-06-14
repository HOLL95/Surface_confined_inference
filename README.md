Python project for surface confined electrochemistry problems - a work in progress! 
# Linux Installation Guide
## Cloning
Obtain the software by running the following command in an appropriate directory
```
git clone https://github.com/HOLL95/Surface_confined_inference/ --recurse-submodules
```
## CVODE
This project depends on many things, most of which are downloaded automatically. However, the C++ ODE solver package CVODE requires user install
To install CVODE please download the software [here](https://computing.llnl.gov/projects/sundials/sundials-software) and follow the installation instructions found [here](https://computing.llnl.gov/projects/sundials/sundials-software), which tells you to make a ```buildir```
Once you have installed CVODE, please tell CMake the location of the buildir, either by editing in your .bashrc file
```
export CVODE_PATH="/absolute/path/to/builddir"
```
and running
```
source .bashrc
```
running the command in your terminal
```
export CVODE_PATH="/absolute/path/to/builddir"
```
or replacing the appropriate line in the ```Surface_confined_inference/C_src/CMakeLists.txt``` file directly.
```
set (SUNDIALS_DIR /path/to/builddir)
```
You can check that the former two approaches have worked by noting the output from
```
echo $CVODE_PATH
```
## Installing
We recommend that you use a virtual environment ([install instructions](https://virtualenv.pypa.io/en/latest/installation.html)). To activate:
```
python -m venv venv
```
```
source venv/bin/activate
```
In the virtual environment, navigate to the Surface_confined_inference directory, and run
```
python -m pip install . -r requirements.txt
```
check that your installation has worked by running
```
python example_usage.py
```
