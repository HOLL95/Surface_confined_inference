import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
def convert_to_bode(spectra):
        spectra=[complex(x, y) for x,y in zip(spectra[:,0], spectra[:,1])]
        phase=np.angle(spectra, deg=True)#np.arctan(np.divide(-spectra[:,1], spectra[:,0]))*(180/math.pi)
        #print(np.divide(spectra[:,1], spectra[:,0]))
        magnitude=np.log10(np.abs(spectra))
        return np.column_stack((phase,magnitude))
class SimpleSurfaceCircuit:
    def __init__(self,**kwargs):
        accepted_c_vals=["C","Q"]
        capacitor_elements=["f", "dl"]
        alphas=[]
        for i in range(0, len(capacitor_elements)):
            elem=capacitor_elements[i]
            celem="C"+elem
            
            accepted_vals=[x+elem for x in accepted_c_vals]
            if celem not in kwargs:
                kwargs[celem]=celem
            elif kwargs[celem] not in accepted_vals:
                raise KeyError("{0} must be one of either {1}, not {2}".format(celem, accepted_vals, kwargs[celem]))
            elif kwargs[celem]=="Q"+elem:
                alphas.append("alpha"+elem)
        self.options=kwargs
        self.parameters=["Rsol", "Rf", kwargs["Cf"], kwargs["Cdl"]]+alphas
        if "normalise" not in kwargs:
            kwargs["normalise"]=False
        self.normalise=kwargs["normalise"]
        if self.normalise==True:
            if "boundaries" not in kwargs:
                raise KeyError("Need to define boundaries if normalisation is required")
            else:
                keys=kwargs["boundaries"].keys()
                for key in self.parameters:
                    if key not in keys:
                        raise KeyError("Boundaries needed for {0}".format(key))
                self.boundaries=kwargs["boundaries"]
    def n_parameters(self):
        return len(self.parameters)
    def n_outputs(self):
        return 2
    def simulate(self, parameters, frequencies, test=False):
        p_dict=dict(zip(self.parameters, parameters))
        if self.normalise==True:
            
            parameters=[sci.un_normalise(p_dict[x], self.boundaries[x]) for x in self.parameters]
            p_dict=dict(zip(self.parameters, parameters))
        freq=np.array(frequencies)
        z_cf=1j*p_dict[self.options["Cf"]]
        if self.options["Cf"]=="Qf":
            omega=np.power(freq, p_dict["alphaf"])
        else:
            omega=freq
        z_cf=np.multiply(omega, z_cf)
        z_cdl=1j*p_dict[self.options["Cdl"]]
        
        if self.options["Cdl"]=="Qdl":
            omega=np.power(freq, p_dict["alphadl"])
        else:
            omega=freq
        z_cdl=np.multiply(omega, z_cdl)
        z_para=z_cdl+(z_cf/(z_cf*p_dict["Rf"]+1))
        imp=np.array(p_dict["Rsol"]+1/z_para)
        return_val=np.column_stack((imp.real,imp.imag))
        return_val=sci.convert_to_bode(return_val)
        if test==True:
            print(p_dict)
        #sci.plot.bode(return_val, frequencies,  data_type="phase_mag")#, frequencies)
        #plt.show()
        return return_val
        
