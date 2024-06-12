from scipy.stats import norm, lognorm, skewnorm, loguniform
import numpy as np
import itertools
import copy
import math
import matplotlib.pyplot as plt
class dispersion:
    def __init__(self, dispersion_bins, dispersion_parameters, dispersion_distributions, optim_list, fixed_parameters={}):
        self.bins=dispersion_bins
        self.dispersion_parameters=dispersion_parameters
        self.distributions=dispersion_distributions
        if len(self.bins)!=len(self.params):
            print(self.bins,self.params)
            raise ValueError("Need to define number of bins for each parameter")
        if len(self.distributions)!=len(self.params):
            print("Dipersion distributions are " ,self.distributions,"dispersion parameters are ",self.params)
            raise ValueError("Need to define distributions for each parameter")
        if type(self.bins) is not list:
            self.bins=[self.bins]
        checking_dict={"normal":["_mean", "_std"], "uniform":["_lower", "_upper"], "lognormal":["_logmean", "_logstd"], "loguniform":["_loglower", "_logupper"]}
        for i in range(0, len(self.params)):
            if self.distributions[i] in checking_dict:
                for distribution_parameter in check_dict[self.distributions[i]]:
                    if self.params[i]+distribution_parameter not in self.optim_list or fixed_parameters:
                        raise ValueError(f"{self.distributions[i]} requires {[self.parmams[i]+x for x in check_dict[self.distributions[i]]]} in either optim_list or fixed_parameters")
            else:
                raise KeyError(self.distributions[i]+" distribution not implemented")
    def generic_dispersion(self, dim_dict, GH_dict=None):
        weight_arrays=[]
        value_arrays=[]
        for i in range(0, len(self.params)):
            if self.distributions[i]=="uniform":
                    value_arrays.append(np.linspace(self.params[i]+"_lower", self.params[i]+"_upper", self.bins[i]))
                    weight_arrays.append([1/self.bins[i]]*self.bins[i])
            elif self.distributions[i]=="normal":
                    param_mean=dim_dict[self.params[i]+"_mean"]
                    param_std=dim_dict[self.params[i]+"_std"]
                    if type(GH_dict) is dict:
                        param_vals=[(param_std*math.sqrt(2)*node)+param_mean for node in GH_dict["nodes"]]
                        param_weights=GH_dict["normal_weights"]
                    else:
                        min_val=norm.ppf(1e-4, loc=param_mean, scale=param_std)
                        max_val=norm.ppf(1-1e-4, loc=param_mean, scale=param_std)
                        param_vals=np.linspace(min_val, max_val, self.bins[i])
                        param_weights=np.zeros(self.bins[i])
                        param_weights[0]=norm.cdf(param_vals[0],loc=param_mean, scale=param_std)
                        param_midpoints=np.zeros(self.bins[i])
                        param_midpoints[0]=norm.ppf((1e-4/2), loc=param_mean, scale=param_std)
                        for j in range(1, self.bins[i]):
                            param_weights[j]=norm.cdf(param_vals[j],loc=param_mean, scale=param_std)-norm.cdf(param_vals[j-1],loc=param_mean, scale=param_std)
                            param_midpoints[j]=(param_vals[j-1]+param_vals[j])/2
                        param_vals=param_midpoints
                    value_arrays.append(param_vals)
                    weight_arrays.append(param_weights)
            elif self.distributions[i]=="lognormal":
                    param_loc=0
                    param_shape=dim_dict[self.params[i]+"_shape"]
                    param_scale=dim_dict[self.params[i]+"_scale"]
                    value_range=np.linspace(1e-4, 1-(1e-4), self.bins[i])
                    #min_val=lognorm.ppf(1e-4, param_shape, loc=param_loc, scale=param_scale)
                    #max_val=lognorm.ppf(1-1e-4, param_shape, loc=param_loc, scale=param_scale)
                    param_vals=np.array([lognorm.ppf(x, param_shape, loc=param_loc, scale=param_scale) for x in value_range])#
                    param_weights=np.zeros(self.bins[i])
                    param_weights[0]=lognorm.cdf(param_vals[0],param_shape, loc=param_loc, scale=param_scale)
                    param_midpoints=np.zeros(self.bins[i])
                    param_midpoints[0]=lognorm.ppf((1e-4/2), param_shape, loc=param_loc, scale=param_scale)
                    for j in range(1, self.bins[i]):
                        param_weights[j]=lognorm.cdf(param_vals[j],param_shape, loc=param_loc, scale=param_scale)-lognorm.cdf(param_vals[j-1],param_shape, loc=param_loc, scale=param_scale)
                        param_midpoints[j]=(param_vals[j-1]+param_vals[j])/2
                    value_arrays.append(param_midpoints)
                    weight_arrays.append(param_weights)
            elif self.distributions[i]=="skewed_normal":
                    param_mean=dim_dict[self.params[i]+"_mean"]
                    param_std=dim_dict[self.params[i]+"_std"]
                    param_skew=dim_dict[self.params[i]+"_skew"]
                    value_range=np.linspace(1e-4, 1-(1e-4), self.bins[i])
                    #param_vals=np.linspace(min_val, max_val, self.bins[i])
                    param_vals=np.array([skewnorm.ppf(x,  param_skew,loc=param_mean, scale=param_std) for x in value_range])#
                    param_weights=np.zeros(self.bins[i])
                    param_weights[0]=skewnorm.cdf(param_vals[0],param_skew, loc=param_mean, scale=param_std)
                    param_midpoints=np.zeros(self.bins[i])
                    param_midpoints[0]=skewnorm.ppf((1e-4/2), param_skew, loc=param_mean, scale=param_std)
                    for j in range(1, self.bins[i]):
                        param_weights[j]=skewnorm.cdf(param_vals[j],param_skew, loc=param_mean, scale=param_std)-skewnorm.cdf(param_vals[j-1],param_skew, loc=param_mean, scale=param_std)
                        param_midpoints[j]=(param_vals[j-1]+param_vals[j])/2
                    value_arrays.append(param_midpoints)
                    weight_arrays.append(param_weights)
            elif self.distributions[i]=="log_uniform":
                    param_upper=dim_dict[self.params[i]+"_logupper"]
                    param_lower=dim_dict[self.params[i]+"_loglower"]
                    min_val=loguniform.ppf(1e-4, param_lower, param_upper, loc=0, scale=1)
                    max_val=loguniform.ppf(1-1e-4, param_lower,param_upper, loc=0, scale=1)
                    param_vals=np.linspace(min_val, max_val, self.bins[i])
                    param_weights=np.zeros(self.bins[i])
                    param_weights[0]=loguniform.cdf(min_val, param_lower, param_upper, loc=0, scale=1)
                    param_midpoints=np.zeros(self.bins[i])
                    param_midpoints[0]=loguniform.ppf((1e-4)/2, param_lower,param_upper, loc=0, scale=1)
                    for j in range(1, self.bins[i]):
                        param_weights[j]=loguniform.cdf(param_vals[j], param_lower, param_upper, loc=0, scale=1)-loguniform.cdf(param_vals[j-1], param_lower, param_upper, loc=0, scale=1)
                        param_midpoints[j]=(param_vals[j-1]+param_vals[j])/2
                    value_arrays.append(param_midpoints)
                    weight_arrays.append(param_weights)
        total_len=np.prod(self.bins)
        weight_combinations=list(itertools.product(*weight_arrays))
        value_combinations=list(itertools.product(*value_arrays))
        sim_params=copy.deepcopy(self.params)
        for i in range(0, len(sim_params)):
            if sim_params[i]=="E0":
                sim_params[i]="E_0"
            if sim_params[i]=="k0":
                sim_params[i]="k_0"
        return sim_params, value_combinations, weight_combinations
