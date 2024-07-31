import numpy as np
import Surface_confined_inference as sci
import re
import matplotlib.pyplot as plt
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
unit_dict={
        "E0": "V",
        'E_start': "V", #(starting dc voltage - V)
        'E_reverse': "V",
        'omega':"Hz",#8.88480830076,  #    (frequency Hz)
        'd_E': "V",   #(ac voltage amplitude - V) freq_range[j],#
        'v': '$V s^{-1}$',   #       (scan rate s^-1)
        'area': '$cm^{2}$', #(electrode surface area cm^2)
        'Ru': "$\\Omega$",  #     (uncompensated resistance ohms)
        'Cdl': "F", #(capacitance parameters)
        'CdlE1': "",#0.000653657774506,
        'CdlE2': "",#0.000245772700637,
        'CdlE3': "",#1.10053945995e-06,
        'gamma': 'mol cm$^{-2}$',
        'k0': '$s^{-1}$', #(reaction rate s-1)
        "kcat":"$s^{-1}$",
        'alpha': "",
        'E0_skew':"",
        "E0_mean":"V",
        "E0_std": "V",
        "k0_shape":"",
        "sampling_freq":"$s^{-1}$",
        "k0_loc":"",
        "k0_scale":"",
        "cap_phase":"",
        'phase' : "",
        "alpha_mean": "",
        "alpha_std": "",
        "":"",
        "noise":"",
        "error":"$\\mu A$",
        "sep":"V",
        "cpe_alpha_faradaic" :"",
        "cpe_alpha_cdl" :"",
        "sigma":"",
        "dcv_sep":"V",
        "SWV_constant":"",
        "SWV_linear":"",
        "SWV_squared":"",
        "SWV_cubed":"",
        }
fancy_names={
        "E0": '$E^0$',
        'E_start': '$E_{start}$', #(starting dc voltage - V)
        'E_reverse': '$E_{reverse}$',
        'omega':'$\\omega$',#8.88480830076,  #    (frequency Hz)
        'd_E': "$\\Delta E$",   #(ac voltage amplitude - V) freq_range[j],#
        'v': "v",   #       (scan rate s^-1)
        'area': "Area", #(electrode surface area cm^2)
        'Ru': "$R_u$",  #     (uncompensated resistance ohms)
        'Cdl': "$C_{dl}$", #(capacitance parameters)
        'CdlE1': "$C_{dlE1}$",#0.000653657774506,
        'CdlE2': "$C_{dlE2}$",#0.000245772700637,
        'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
        'gamma': '$\\Gamma$',
        'E0_skew':"$E^0$ skew",
        'k0': '$k_0$', #(reaction rate s-1)
        'alpha': "$\\alpha$",
        "E0_mean":"$E^0 \\mu$",
        "E0_std": "$E^0 \\sigma$",
        "cap_phase":"C$_{dl}$ $\\eta$",
        "k0_shape":"$\\log(k^0) \\sigma$",
        "k0_scale":"$\\log(k^0) \\mu$",
        "alpha_mean": "$\\alpha\\mu$",
        "alpha_std": "$\\alpha\\sigma$",
        'phase' : "$\\eta$",
        "kcat":"$k_{cat}$",
        "sampling_freq":"Sampling rate",
        "":"Experiment",
        "noise":"$\\sigma$",
        "error":"RMSE",
        "sep":"Seperation",
        "cpe_alpha_faradaic" :"$\\psi$ $(\\Gamma)$",
        "cpe_alpha_cdl" :"$\\psi$ $(C_{dl})$",
        "sigma":"$\\sigma$",
        "dcv_sep":"Separation",
        "SWV_constant":"$I_b$",
        "SWV_linear":"$I_b^1$",
        "SWV_squared":"$I_b^2$",
        "SWV_cubed":"$I_b^3$",
        }
dispersion_symbols={
                    "mean":"$\\mu$", 
                    "std":"$\\sigma$", 
                    "skew":"$\\kappa$", 
                    "shape":"$\\mu$",
                    "scale":"$\\sigma$",
                    "upper":r"${_ub}$", 
                    "lower":r"${_lb}$", 
                    "logupper":r"${_ub}$", 
                    "loglower":r"${_lb}$", 
            }
nounits=["skew", "shape", "scale", "logupper", "loglower"]
dispersion_param_names=list(dispersion_symbols.keys())
def det_subplots( value):
    if np.floor(np.sqrt(value))**2==value:
        return int(np.sqrt(value)), int(np.sqrt(value))
    elif value<4:
        return 1, value
    if value<=10:
        start_val=2
    else:
        start_val=3

    rows=range(start_val, int(np.ceil(value/start_val)))
    for i in range(0, 10):
        modulos=np.array([value%x for x in rows])
        idx_0=(np.where(modulos==0))
        if len(idx_0[0])!=0:
            return int(rows[idx_0[0][-1]]), int(value/rows[idx_0[0][-1]])
        value+=1
def get_titles(titles, **kwargs):

    if "units" not in kwargs:
        kwargs["units"]=True
    if "positions" not in kwargs:
        kwargs["positions"]=range(0, len(titles))
    params=["" for x in kwargs["positions"]]
    plot_units={}
    for i in range(0, len(kwargs["positions"])):
        z=kwargs["positions"][i]
        if titles[z] in sci._utils.fancy_names:
            plot_units[titles[z]]=sci._utils.unit_dict[titles[z]]
            if kwargs["units"]==True and sci._utils.unit_dict[titles[z]]!="":
                
                params[i]=sci._utils.fancy_names[titles[z]]+" ("+sci._utils.unit_dict[titles[z]]+")" 
                
            else:
                params[i]=sci._utils.fancy_names[titles[z]]
        else:
            disped=False
            
            for q in range(0, len(sci._utils.dispersion_param_names)):
                if sci._utils.dispersion_param_names[q] in sci._utils.nounits:
                    nounits=True
                else:
                    nounits=False
                if "_"+sci._utils.dispersion_param_names[q] in titles[z]:
                    disped=True
                    if re.search(".*_[1-9]+", titles[z]) is not None:
                        fname, true_name=sci._utils.numbered_title(titles[z], units=False)
                    else:
                        fname=re.findall(".*(?=_{0})".format(sci._utils.dispersion_param_names[q]), titles[z])[0]
                        true_name=fname
                    if nounits==True:
                        plot_units[titles[z]]=""
                    else:
                        plot_units[titles[z]]=sci._utils.unit_dict[true_name]
                    if kwargs["units"]==True and sci._utils.unit_dict[true_name]!="" and nounits==False:
                        params[i]=sci._utils.fancy_names[fname]+sci._utils.dispersion_symbols[sci._utils.dispersion_param_names[q]] +" ("+sci._utils.unit_dict[true_name]+")" 
                    else:
                        params[i]=sci._utils.fancy_names[fname]+sci._utils.dispersion_symbols[sci._utils.dispersion_param_names[q]]
            if disped==False:
                if re.search(".*_[1-9]+", titles[z]) is not None:
                    params[i],true_name=sci._utils.numbered_title(titles[z], units=kwargs["units"])
                    plot_units[titles[z]]=sci._utils.unit_dict[true_name]
                else:
                    params[i]=titles[z]
                

            
    return params
def numbered_title( name, **kwargs):
    for key in sci._utils.fancy_names.keys():
        underscore_idx=[i for i in range(0, len(name)) if name[i]=="_"]
        true_name=name[:underscore_idx[0]]
        if len(underscore_idx)==1:
            value=name[underscore_idx[0]+1:]
        else:
            value=name[underscore_idx[0]+1:underscore_idx[1]]
        if kwargs["units"]==True and sci._utils.unit_dict[true_name]!="":

            return_name=sci._utils.fancy_names[true_name]+"$_{"+value +"}$"+" ("+sci._utils.unit_dict[true_name]+")" 
        else:
            return_name=sci._utils.fancy_names[true_name]+"$_{"+value +"}$"
        break
    return return_name, true_name
def format_values( value, dp=2):
    abs_val=abs(value)
    if abs_val<1000 and abs_val>0.015:
        return str(round(value, dp))
    else:
        return "{:.{}e}".format(value, dp)
