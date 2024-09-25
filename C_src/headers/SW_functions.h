#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace std;
double SW_potential(int j, double SF,double dE, double Esw, double E_start, int scan_direction);
py::object SW_current(std::vector<double> times, std::unordered_map<std::string, double> params);