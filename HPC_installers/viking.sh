module load Python/3.11.5-GCCcore-13.2.0
module load pybind11/2.11.1-GCCcore-12.3.0
module load SUNDIALS/6.6.0-foss-2023a
alias python="python3.11"
python -m pip install -r requirements.txt
export IN_VIKING="true"
