#include <cvode/cvode.h>            /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sundials/sundials_types.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace std;
extern "C" int single_e(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);
void updateCdlp(std::unordered_map<std::string, double>& params, double Cdlp);
extern "C" int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
template<typename PhaseFunc>
double mono_E(const std::unordered_map<std::string, double>& params, double t, PhaseFunc phase_func){
	double phase = phase_func(params, t);
	
	double E_dc;
	double E_t;
	if (t<params.at("tr")){
		E_dc=params.at("E_start")+(params.at("v")*t); 
	}else {
		E_dc=params.at("E_reverse")-(params.at("v")*(t-params.at("tr")));
	}
	E_t= E_dc+(params.at("delta_E")*(std::sin((params.at("omega")*t)+phase)));
	//std::cout<<"headerE:"<<E_t<<"headert: "<<t<<" ";
	return E_t;
}


std::vector<double> potential(const std::vector<double>& times,const std::unordered_map<std::string, double>& params);
double mono_dE(const std::unordered_map<std::string, double>& params, double t, double phase);
double mono_dE_sine_phase(const std::unordered_map<std::string, double>& params, double t);
double Marcus_kinetics_oxidation(const std::unordered_map<std::string, double>& params, double Er);
double Marcus_kinetics_reduction(const std::unordered_map<std::string, double>& params, double Er);
double BV_oxidation(std::unordered_map<std::string, double>& params, double Er);
double BV_reduction(std::unordered_map<std::string, double>& params, double Er);
