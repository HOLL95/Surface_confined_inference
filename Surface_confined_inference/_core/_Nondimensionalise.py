import Surface_confined_inference as sci
import copy
import re
from numpy import pi


class NDParams:
    def __init__(self, experiment_type, input_parameters):

        self.F = 96485.3328959
        self.R = 8.314459848
        self.T = input_parameters["Temp"]
        self.c_E0 = (self.R * self.T) / self.F
        V_nondim = ["FTACV", "DCV", "TrumpetPlot"]
        Omega_nondim = ["PSV", "SquareWave"]
        K_nondim = ["SquareWave"]
        if experiment_type in V_nondim:
            time_constant = input_parameters["v"]
        elif experiment_type in Omega_nondim:
            time_constant = input_parameters["omega"]*self.c_E0
        self.c_T0 = abs(self.c_E0 / time_constant)
        if experiment_type in K_nondim:
            self.c_T0 = 1 / input_parameters["omega"]
        self.c_I0 = (
            self.F * input_parameters["area"] * input_parameters["Surface_coverage"]
        ) / self.c_T0
        self.c_Gamma = input_parameters["Surface_coverage"]
        self.area = input_parameters["area"]

    def construct_function_dict(self, dim_dict):

        function_dict = {}
        for key in dim_dict:
            if key[0] == "E" or key == "delta_E":
                function_dict[key] = self.e_nondim
            elif key[0] == "k" and "scale" not in key:
                function_dict[key] = self.t_nondim
            elif key == "Ru":
                function_dict[key] = self.Ru_nondim
            elif key == "Cdl":
                function_dict[key] = self.Cdl_nondim
            elif key == "gamma":
                function_dict[key] = self.gamma_nondim
            elif key == "omega":
                function_dict[key] = self.omega_nondim
            elif key == "v":
                function_dict[key] = self.v_nondim
            else:
                function_dict[key] = lambda x:x
        function_dict["cap_phase"] = lambda x:x
        self.function_dict = function_dict
    def construct_function_dict_SW(self, dim_dict):

        function_dict = {}
        for key in dim_dict:
            if key[0] == "k" and "scale" not in key:
                function_dict[key] = self.t_nondim
            elif key == "gamma":
                function_dict[key] = self.gamma_nondim
            else:
                function_dict[key] = lambda x:x
        function_dict["cap_phase"] = lambda x:x
        self.function_dict = function_dict
    def redimensionalise(self, nondim_dict):

        dim_dict = {}
        for key in nondim_dict:
            if key[0] == "E" or key == "delta_E":
                dim_dict[key] = self.e_redim(nondim_dict[key])
            elif key[0] == "k" and "scale" not in key:
                dim_dict[key] = self.t_redim(nondim_dict[key])
            elif key == "Ru":
                dim_dict[key] = self.Ru_redim(nondim_dict[key])
            elif key == "Cdl":
                dim_dict[key] = self.Cdl_redim(nondim_dict[key])
            elif key == "gamma":
                dim_dict[key] = self.gamma_redim(nondim_dict[key])
            elif key == "omega":
                dim_dict[key] = self.omega_redim(nondim_dict[key])
            elif key == "v":
                dim_dict[key] = self.v_redim(nondim_dict[key])
            else:
                dim_dict[key] = nondim_dict[key]
        return dim_dict
       

    def e_nondim(self, value):
        return value / self.c_E0

    def t_nondim(self, value):
        return value * self.c_T0

    def i_nondim(self, value):
        return value / self.c_I0

    def Ru_nondim(self, value):
        return( value / self.c_E0 )* self.c_I0

    def Cdl_nondim(self, value):
        return value / self.c_I0 / self.c_T0 * (self.area * self.c_E0)

    def gamma_nondim(self, value):
        return value / self.c_Gamma

    def omega_nondim(self, value):
        return value * (2 * pi * self.c_T0)

    def v_nondim(self, value):
        return value * self.c_T0 / self.c_E0
    def e_redim(self, value):
        return value * self.c_E0

    def t_redim(self, value):
        return value / self.c_T0

    def Ru_redim(self, value):
        return (value * self.c_E0) / self.c_I0

    def Cdl_redim(self, value):
        return value* (self.c_I0 * self.c_T0)  / (self.area * self.c_E0)

    def gamma_redim(self, value):
        return value * self.c_Gamma

    def omega_redim(self, value):
        return value / (2 * pi * self.c_T0)

    def v_redim(self, value):
        return value / self.c_T0 * self.c_E0

    
