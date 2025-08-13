
#include <stdio.h>
#include <iostream> 
#include <cvode/cvode.h>            /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sundials/sundials_types.h>
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <string>
#include "functions.h"
using namespace std;

#define Ith(v, i) NV_Ith_S(v, i - 1) /* i-th vector component i=1..NEQ */
#define IJth(A, i, j) \
  SM_ELEMENT_D(A, i - 1, j - 1) /* (i,j)-th matrix component i,j=1..NEQ */


//TODO rewrite as lambdas 

extern "C" int multi_e(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
    std::unordered_map<std::string, double>* params = static_cast<std::unordered_map<std::string, double>*>(user_data);
    double I = Ith(y, 1);
    double dIdt, Cdlp;
    double Er, dE, cap_Er, cap_dE;
    double dQdt, dQminusdt, dQ2minusdt, dQHplusdt, dQHdt, dQHminusdt, dQH22plusdt, dQH21plusdt, dchargedt;
    double Q = Ith(y, 2);
    double Qminus = Ith(y, 3);
    double Q2minus = Ith(y, 4);// Jacobian Matrix Elements with Simplified Notation
    double QHplus = Ith(y, 5);
    double QH22plus = Ith(y, 6);
    double QH = Ith(y, 7);
    double QHminus = Ith(y, 8);
    double QH21plus = Ith(y, 9);
    double QH2=(-Q - Qminus - Q2minus - QH - QHplus - QHminus - QH21plus - QH22plus + 1);

    Er=mono_E(*params, t, (*params)["phase"])-I*(*params)["Ru"];
    cap_Er=mono_E(*params, t,(*params)["cap_phase"])-I*(*params)["Ru"];
    cap_dE=mono_dE(*params, t, (*params)["cap_phase"]);



    for (int j = 1; j < 7; j++) {
        string k = "k0_" + std::to_string(j);
        string a = "alpha_" + std::to_string(j);
        string e = "E0_" + std::to_string(j);
        (*params)["kred_" + std::to_string(j)] = params->at(k) * std::exp(-params->at(a) * (Er - params->at(e)));
        (*params)["kox_" + std::to_string(j)] = params->at(k) * std::exp((1 - params->at(a)) * (Er - params->at(e)));
    }

    

    dQdt = -Q*params->at("kred_1") - Q*params->at("kp_i") + Qminus*params->at("kox_1") + QHplus*params->at("kd_i");
    dQminusdt = Q*params->at("kred_1") - Qminus*params->at("kox_1") - Qminus*params->at("kred_2") - Qminus*params->at("kp_ii") + Q2minus*params->at("kox_2") + QH*params->at("kd_ii");
    dQ2minusdt = Qminus*params->at("kred_2") - Q2minus*params->at("kox_2") - Q2minus*params->at("kp_iii") + QHminus*params->at("kd_iii");
    dQHplusdt = Q*params->at("kp_i") + QH*params->at("kox_3") - QHplus*params->at("kred_3") - QHplus*params->at("kd_i") - QHplus*params->at("kp_iv") + QH22plus*params->at("kd_iv");
    dQHdt = Qminus*params->at("kp_ii") - QH*params->at("kox_3") - QH*params->at("kred_4") - QH*params->at("kd_ii") - QH*params->at("kp_v") + QHplus*params->at("kred_3") + QHminus*params->at("kox_4") + QH21plus*params->at("kd_v");
    dQHminusdt = Q2minus*params->at("kp_iii") + QH*params->at("kred_4") - QHminus*params->at("kox_4") - QHminus*params->at("kd_iii") - QHminus*params->at("kp_vi") + params->at("kd_vi")*QH2;
    dQH22plusdt = QHplus*params->at("kp_iv") + QH21plus*params->at("kox_5") - QH22plus*params->at("kred_5") - QH22plus*params->at("kd_iv");
    dQH21plusdt = QH*params->at("kp_v") - QH21plus*params->at("kox_5") - QH21plus*params->at("kred_6") - QH21plus*params->at("kd_v") + QH22plus*params->at("kred_5") + params->at("kox_6")*QH2;
    
    Ith(ydot, 2) = dQdt;
    Ith(ydot, 3) = dQminusdt;
    Ith(ydot, 4) = dQ2minusdt;
    Ith(ydot, 5) = dQHplusdt;
    Ith(ydot, 6) = dQH22plusdt;
    Ith(ydot, 7) = dQHdt;
    Ith(ydot, 8) = dQHminusdt;
    Ith(ydot, 9) = dQH21plusdt;
    dchargedt=Q*params->at("kred_1") - \
                Qminus*params->at("kox_1") + \
                Qminus*params->at("kred_2") - \
                Q2minus*params->at("kox_2") - \
                QH*params->at("kox_3") + \
                QH*params->at("kred_4") + \
                QHplus*params->at("kred_3") - \
                QHminus*params->at("kox_4") - \
                QH21plus*params->at("kox_5") + \
                QH21plus*params->at("kred_6") + \
                QH22plus*params->at("kred_5") - \
                params->at("kox_6")*QH2;
    
    double Er2=pow(cap_Er,2);
    Cdlp=(*params)["Cdl"]*(1+((*params)["CdlE1"]*cap_Er)+((*params)["CdlE2"]*Er2)+((*params)["CdlE3"]*Er2*cap_Er));
    dIdt=-(1/((*params)["Ru"]*Cdlp))*(I-(*params)["gamma"]*dchargedt-Cdlp*cap_dE);
    


    Ith(ydot, 1) = dIdt;
    updateCdlp(*params, Cdlp);

    return (0);
    }

extern "C" int Jac_multi_e(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    auto params = static_cast<std::unordered_map<std::string, double>*>(user_data);

    // Extract state variables
    double I = Ith(y, 1);
    double Q = Ith(y, 2);
    double Qminus = Ith(y, 3);
    double Q2minus = Ith(y, 4);
    double QHplus = Ith(y, 5);
    double QH22plus = Ith(y, 6);
    double QH = Ith(y, 7);
    double QHminus = Ith(y, 8);
    double QH21plus = Ith(y, 9);

    double QH2 = (-Q - Qminus - Q2minus - QH - QHplus - QHminus - QH21plus - QH22plus + 1);
    double Cdl=params->at("Cdl");
    double R =params->at("Ru");
    double CdlR=Cdl*R;
    double kred_1=params->at("kred_1");
    double kred_2=params->at("kred_2");
    double kred_3=params->at("kred_3");
    double kred_4=params->at("kred_4");
    double kred_5=params->at("kred_5");
    double kred_6=params->at("kred_6");
    double kox_1=params->at("kox_1");
    double kox_2=params->at("kox_2");
    double kox_3=params->at("kox_3");
    double kox_4=params->at("kox_4");
    double kox_5=params->at("kox_5");
    double kox_6=params->at("kox_6");
    double alpha_1=params->at("alpha_1");
    double alpha_2=params->at("alpha_2");
    double alpha_3=params->at("alpha_3");
    double alpha_4=params->at("alpha_4");
    double alpha_5=params->at("alpha_5");
    double alpha_6=params->at("alpha_6");
    double kp_i=params->at("kp_i");
    double kp_ii=params->at("kp_ii");
    double kp_iii=params->at("kp_iii");
    double kp_iv=params->at("kp_iv");
    double kp_v=params->at("kp_v");
    double kp_vi=params->at("kp_vi");
    double kd_i=params->at("kd_i");
    double kd_ii=params->at("kd_ii");
    double kd_iii=params->at("kd_iii");
    double kd_iv=params->at("kd_iv");
    double kd_v=params->at("kd_v");
    double kd_vi=params->at("kd_vi");
    double gamma = params->at("gamma");
    
    // k0_x*exp(-alpha_x*(E - E0_x - I*R)) replaced with kred_x
    // k0_x*exp((1-alpha_x)*(E - E0_x - I*R)) replaced with kox_x
    IJth(J, 1, 1) = gamma*(Q*R*alpha_1*kred_1 + Qminus*R*alpha_2*kred_2 + Qminus*R*kox_1*(1 - alpha_1) + Q2minus*R*kox_2*(1 - alpha_2) + QH*R*alpha_4*kred_4 + QH*R*kox_3*(1 - alpha_3) + QHplus*R*alpha_3*kred_3 + QHminus*R*kox_4*(1 - alpha_4) + QH21plus*R*alpha_6*kred_6 + QH21plus*R*kox_5*(1 - alpha_5) + QH22plus*R*alpha_5*kred_5 + R*kox_6*(1 - alpha_6)*QH2)/CdlR - 1/CdlR; // dI/dI
    IJth(J, 1, 2) = gamma*(kred_1 + kox_6)/CdlR; // dI/dQ
    IJth(J, 1, 3) = gamma*(-kox_1 + kred_2 + kox_6)/CdlR; // dI/dQminus
    IJth(J, 1, 4) = gamma*(-kox_2 + kox_6)/CdlR; // dI/dQ2minus
    IJth(J, 1, 5) = gamma*(kred_3 + kox_6)/CdlR; // dI/dQHplus
    IJth(J, 1, 6) = gamma*(kred_5 + kox_6)/CdlR; // dI/dQH22plus
    IJth(J, 1, 7) = gamma*(-kox_3 + kred_4 + kox_6)/CdlR; // dI/dQH
    IJth(J, 1, 8) = gamma*(-kox_4 + kox_6)/CdlR; // dI/dQHminus
    IJth(J, 1, 9) = gamma*(-kox_5 + kox_6 + kred_6)/CdlR; // dI/dQH21plus
    
    IJth(J, 2, 1) = -Q*R*alpha_1*kred_1 - Qminus*R*kox_1*(1 - alpha_1); // dQ/dI
    IJth(J, 2, 2) = -kred_1 - kp_i; // dQ/dQ
    IJth(J, 2, 3) = kox_1; // dQ/dQminus
    IJth(J, 2, 4) = 0; // dQ/dQ2minus
    IJth(J, 2, 5) = kd_i; // dQ/dQHplus
    IJth(J, 2, 6) = 0; // dQ/dQH22plus
    IJth(J, 2, 7) = 0; // dQ/dQH
    IJth(J, 2, 8) = 0; // dQ/dQHminus
    IJth(J, 2, 9) = 0; // dQ/dQH21plus

    IJth(J, 3, 1) = Q*R*alpha_1*kred_1 - Qminus*R*alpha_2*kred_2 + Qminus*R*kox_1*(1 - alpha_1) - Q2minus*R*kox_2*(1 - alpha_2); // dQminus/dI
    IJth(J, 3, 2) = kred_1; // dQminus/dQ
    IJth(J, 3, 3) = -kox_1 - kred_2 - kp_ii; // dQminus/dQminus
    IJth(J, 3, 4) = kox_2; // dQminus/dQ2minus
    IJth(J, 3, 5) = 0; // dQminus/dQHplus
    IJth(J, 3, 6) = 0; // dQminus/dQH22plus
    IJth(J, 3, 7) = kd_ii; // dQminus/dQH
    IJth(J, 3, 8) = 0; // dQminus/dQHminus
    IJth(J, 3, 9) = 0; // dQminus/dQH21plus

    IJth(J, 4, 1) = Qminus*R*alpha_2*kred_2 + Q2minus*R*kox_2*(1 - alpha_2); // dQ2minus/dI
    IJth(J, 4, 2) = 0; // dQ2minus/dQ
    IJth(J, 4, 3) = kred_2; // dQ2minus/dQminus
    IJth(J, 4, 4) = -kox_2 - kp_iii; // dQ2minus/dQ2minus
    IJth(J, 4, 5) = 0; // dQ2minus/dQHplus
    IJth(J, 4, 6) = 0; // dQ2minus/dQH22plus
    IJth(J, 4, 7) = 0; // dQ2minus/dQH
    IJth(J, 4, 8) = kd_iii; // dQ2minus/dQHminus
    IJth(J, 4, 9) = 0; // dQ2minus/dQH21plus

    IJth(J, 5, 1) = -QH*R*kox_3*(1 - alpha_3) - QHplus*R*alpha_3*kred_3; // dQHplus/dI
    IJth(J, 5, 2) = kp_i; // dQHplus/dQ
    IJth(J, 5, 3) = 0; // dQHplus/dQminus
    IJth(J, 5, 4) = 0; // dQHplus/dQ2minus
    IJth(J, 5, 5) = -kred_3 - kd_i - kp_iv; // dQHplus/dQHplus
    IJth(J, 5, 6) = kd_iv; // dQHplus/dQH22plus
    IJth(J, 5, 7) = kox_3; // dQHplus/dQH
    IJth(J, 5, 8) = 0; // dQHplus/dQHminus
    IJth(J, 5, 9) = 0; // dQHplus/dQH21plus

    IJth(J, 6, 1) = -QH21plus*R*kox_5*(1 - alpha_5) - QH22plus*R*alpha_5*kred_5; // dQH22plus/dI
    IJth(J, 6, 2) = 0; // dQH22plus/dQ
    IJth(J, 6, 3) = 0; // dQH22plus/dQminus
    IJth(J, 6, 4) = 0; // dQH22plus/dQ2minus
    IJth(J, 6, 5) = kp_iv; // dQH22plus/dQHplus
    IJth(J, 6, 6) = -kred_5 - kd_iv; // dQH22plus/dQH22plus
    IJth(J, 6, 7) = 0; // dQH22plus/dQH
    IJth(J, 6, 8) = 0; // dQH22plus/dQHminus
    IJth(J, 6, 9) = kox_5; // dQH22plus/dQH21plus

    IJth(J, 7, 1) = -QH*R*alpha_4*kred_4 + QH*R*kox_3*(1 - alpha_3) + QHplus*R*alpha_3*kred_3 - QHminus*R*kox_4*(1 - alpha_4); // dQH/dI
    IJth(J, 7, 2) = 0; // dQH/dQ
    IJth(J, 7, 3) = kp_ii; // dQH/dQminus
    IJth(J, 7, 4) = 0; // dQH/dQ2minus
    IJth(J, 7, 5) = kred_3; // dQH/dQHplus
    IJth(J, 7, 6) = 0; // dQH/dQH22plus
    IJth(J, 7, 7) = -kox_3 - kred_4 - kd_ii - kp_v; // dQH/dQH
    IJth(J, 7, 8) = kox_4; // dQH/dQHminus
    IJth(J, 7, 9) = kd_v; // dQH/dQH21plus

    IJth(J, 8, 1) = QH*R*alpha_4*kred_4 + QHminus*R*kox_4*(1 - alpha_4); // dQHminus/dI
    IJth(J, 8, 2) = -kd_vi; // dQHminus/dQ
    IJth(J, 8, 3) = -kd_vi; // dQHminus/dQminus
    IJth(J, 8, 4) = -kd_vi + kp_iii; // dQHminus/dQ2minus
    IJth(J, 8, 5) = -kd_vi; // dQHminus/dQHplus
    IJth(J, 8, 6) = -kd_vi; // dQHminus/dQH22plus
    IJth(J, 8, 7) = kred_4 - kd_vi; // dQHminus/dQH
    IJth(J, 8, 8) = -kox_4 - kd_iii - kd_vi - kp_vi; // dQHminus/dQHminus
    IJth(J, 8, 9) = -kd_vi; // dQHminus/dQH21plus

    IJth(J, 9, 1) = -QH21plus*R*alpha_6*kred_6 + QH21plus*R*kox_5*(1 - alpha_5) + QH22plus*R*alpha_5*kred_5 - R*kox_6*(1 - alpha_6)*QH2; // dQH21plus/dI
    IJth(J, 9, 2) = -kox_6; // dQH21plus/dQ
    IJth(J, 9, 3) = -kox_6; // dQH21plus/dQminus
    IJth(J, 9, 4) = -kox_6; // dQH21plus/dQ2minus
    IJth(J, 9, 5) = -kox_6; // dQH21plus/dQHplus
    IJth(J, 9, 6) = kred_5 - kox_6; // dQH21plus/dQH22plus
    IJth(J, 9, 7) = -kox_6 + kp_v; // dQH21plus/dQH
    IJth(J, 9, 8) = -kox_6; // dQH21plus/dQHminus
    IJth(J, 9, 9) = -kox_5 - kox_6 - kred_6 - kd_v; // dQH21plus/dQH21plus

   
    return (0);
}

