
#include <stdio.h>
#include <iostream> 
#include <cvode/cvode.h>            /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sundials/sundials_types.h>
#include <pybind11/pybind11.h>
#include "functions.h"
using namespace std;

#define Ith(v, i) NV_Ith_S(v, i - 1) /* i-th vector component i=1..NEQ */
#define IJth(A, i, j) \
  SM_ELEMENT_D(A, i - 1, j - 1) /* (i,j)-th matrix component i,j=1..NEQ */


//TODO rewrite as lambdas 

extern "C" int single_e(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
  std::unordered_map<std::string, double>* params = static_cast<std::unordered_map<std::string, double>*>(user_data);
  double I = Ith(y, 1);
  double theta = Ith(y, 2);
  double kox, kred, dIdt, Cdlp;
  double Er, dE, cap_Er, cap_dE;
  

    
    Er=mono_E(*params, t, (*params)["phase"])-I*(*params)["Ru"];
    cap_Er=mono_E(*params, t,(*params)["cap_phase"])-I*(*params)["Ru"];
    cap_dE=mono_dE(*params, t, (*params)["cap_phase"]);


  
  
  if ((*params)["Marcus_flag"]==1){
    kred=Marcus_kinetics_reduction(*params, Er);
    kox=Marcus_kinetics_oxidation(*params, Er);
  }
  else{
    kred=BV_reduction(*params, Er);
    kox=BV_oxidation(*params, Er);
  }
  
  double dtheta = Ith(ydot, 2) =kox*(1-theta)-kred*theta ;
  double Er2=pow(cap_Er,2);
  Cdlp=(*params)["Cdl"]*(1+((*params)["CdlE1"]*cap_Er)+((*params)["CdlE2"]*Er2)+((*params)["CdlE3"]*Er2*cap_Er));
  if (((*params)["Cdl"]<=0)){
    dE=mono_dE(*params, t, (*params)["phase"]);
    dIdt=dE*(1-(*params)["alpha"])*kox*(1-theta)-kred*theta*-(*params)["alpha"]*dE;
  }else{
     dIdt=-(1/((*params)["Ru"]*Cdlp))*(I-(*params)["gamma"]*dtheta-Cdlp*cap_dE);
  }
 

  Ith(ydot, 1) = dIdt;
  updateCdlp(*params, Cdlp);
  
  return (0);
}
void updateCdlp(std::unordered_map<std::string, double>& params, double Cdlp) {
  params["Cdlp"]=Cdlp;
}
extern "C" int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  auto params = static_cast<std::unordered_map<std::string, double>*>(user_data);

  double I = Ith(y, 1);
  double theta = Ith(y, 2);
  double dtheta_dtheta=-(*params)["kox"]-(*params)["kred"];
  double dtheta_dI=-(*params)["Ru"]*theta*(*params)["kred"]-(*params)["Ru"]*(1-(*params)["alpha"])*(1-theta)*(*params)["kox"];
  double dE=mono_dE(*params, t, (*params)["phase"]);
  
  if (((*params)["Cdl"]<=0)){

    IJth(J, 1, 1) = std::pow((1 - (*params)["alpha"]),2) * (*params)["Ru"] * (1 - theta) * dE * (*params)["kox"] - std::pow((*params)["alpha"],2) *  (*params)["Ru"] * theta * dE *(*params)["kred"];
    IJth(J, 1, 2) = -dE*(1-(*params)["alpha"])*(*params)["kox"]-(*params)["kred"]*-(*params)["alpha"]*dE;
    IJth(J, 2, 1) = dtheta_dI;

    IJth(J, 2, 2) = dtheta_dtheta;

  }
  else{
    IJth(J, 1, 1) = -(1/((*params)["Ru"]*(*params)["Cdlp"]))*(1-(*params)["gamma"]*dtheta_dI);
    IJth(J, 1, 2) = -(1/((*params)["Ru"]*(*params)["Cdlp"]))*((*params)["gamma"]*dtheta_dtheta);
    IJth(J, 2, 1) = dtheta_dI;

    IJth(J, 2, 2) = dtheta_dtheta;
  }
 
  return (0);
}

/*
double E1 = mono_E(*params, t, [](const auto& p, double t) { return p.at("phase"); });

double E2 = mono_E(*params, t, [](const auto& p, double t) { 
    return std::sin(p.at("phase_omega") * t + p.at("phase_offset")); 

})
  
;*/

double mono_dE(const std::unordered_map<std::string, double>& params, double t, double phase){ //
	double E_dc;
	if (t < params.at("tr")){
		 E_dc=params.at("v");
	}else {
		 E_dc=-params.at("v");
	}
	return E_dc+(params.at("delta_E")*params.at("omega")*std::cos(params.at("omega")*t+phase));
}
double mono_E(const std::unordered_map<std::string, double>& params, double t, double phase){
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

std::vector<double> potential(const std::vector<double>& times,const std::unordered_map<std::string, double>& params){
  vector<double> potential_values;
  int num_times=times.size();
  potential_values.resize(num_times);
  for(int i=0; i<num_times; i++){
    potential_values[i]=mono_E(params, times[i], params.at("phase"));
  }

return potential_values;
}
double Marcus_kinetics_oxidation(const std::unordered_map<std::string, double>& params, double Er){
  return 0;
}
double Marcus_kinetics_reduction(const std::unordered_map<std::string, double>& params, double Er){
  return 0;
}
double BV_oxidation(std::unordered_map<std::string, double>& params, double Er){
  double kox= params.at("k0")*std::exp((1-params.at("alpha"))*(Er-params.at("E0")));
  params["kox"]=kox;
  return kox;
}
double BV_reduction(std::unordered_map<std::string, double>& params, double Er){
  double kred=params.at("k0")*std::exp(-params.at("alpha")*(Er-params.at("E0")));
  params["kred"]=kred;
  return kred;
}