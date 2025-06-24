git pull origin master
module load Python/3.11.3-GCCcore-12.3.0
module load pybind11/2.11.1-GCCcore-12.3.0
module load SUNDIALS/6.3.0-foss-2021b
alias python="python3.11"
export IN_ARC="true"
python -m pip install -e .
