module purge OpenSSL/1.1.1w-GCCcore-12.3.0
git pull origin master
module load Python/3.11.3-GCCcore-12.3.0
find $EBROOTPYTHON -name "Python.h" 2>/dev/null
module load pybind11/2.11.1-GCCcore-12.3.0
module load SUNDIALS/6.3.0-foss-2021b
module unload OpenSSL/1.1.1w-GCCcore-12.3.0
alias python="python3.11"
python -m ensurepip --user
python -m pip install --no-cache-dir torch 
export IN_ARC="true"
find $EBROOTPYTHON -name "Python.h" 2>/dev/null
echo $IN_ARC
python -m pip install -e .
