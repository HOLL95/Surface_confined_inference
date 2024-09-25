#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <math.h>
#include <iostream>
#include "SW_functions.h"
namespace py = pybind11;
using namespace std;




double SW_potential(int j, double SF, double dE, double Esw, double E_start, int scan_direction){
  double first_term=ceil((j/(SF/2))*0.5)*dE;
  int second_term;
  if (std::ceil(j/(SF/2))/2==std::ceil((j/(SF/2))*0.5)){
    second_term=1;
  }else{
    second_term=-1;
  }
  return E_start+scan_direction*(((first_term+second_term*Esw)+Esw)-dE);
}

py::object SW_current(std::vector<double> times, std::unordered_map<std::string, double> params) {
    double E;
    double numerator, denominator;
    double Itot_sum;
    const double F=96485.3328959;
    const double R=8.314459848;
    const double T = params.at("Temp");
    const double n = params.at("N_elec");
    const double pi = 3.14159265358979323846;
    const double FRT = (n*F)/(R*T);
    const double k0 = params.at("k0");
    const double alpha = params.at("alpha");
    const double gamma = params.at("gamma");
    const double E0 = params.at("E0");
    const double Es = params.at("E_start");
    const double d_E = params.at("scan_increment");
    const double delta_E = params.at("delta_E");
    const int SF = static_cast<int>(params.at("sampling_factor"));
    const double Esw = params.at("SW_amplitude");
    const double v = params.at("v");
    const double E_start=Es-E0;
    int end =(delta_E/d_E)*SF;
    std::vector<double> Itot(end, 0);
    E=FRT*(Es);
    Itot[0]=k0*exp(-alpha*E)/(1+(1+exp(E))*((k0*exp(-alpha*E))/SF));
    Itot_sum=Itot[0];
    for (int j = 1; j <= end; j++) {
      E=FRT*(SW_potential(j, SF, d_E, Esw, E_start, v));
      numerator=k0*exp(-alpha*E)*(1-((1+exp(E))/SF)*Itot_sum);
      denominator=1+((k0*exp(-alpha*E)/SF)*(1+exp(E)));
      Itot[j-1]=numerator/denominator;
      Itot_sum+=Itot[j-1];
    }
    for (int j=0; j<end; j++){
      Itot[j]=Itot[j]*gamma;

    }
    return py::cast(Itot);
  }



