module load git/2.41.0-GCCcore-12.3.0-nodocs
git pull origin master
module load Python/3.11
alias python="python3.11"
module load pybind11/2.11.1-GCCcore-12.3.0
module load SUNDIALS/6.3.0-foss-2021b
python -m ensurepip --user
export IN_ARC="true"
python -m pip install -e .
