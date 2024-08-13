import numpy as np
import Surface_confined_inference as sci
import matplotlib.pyplot as plt
class HeuristicMethod:
    def __init__(self,file, **kwargs):
        if "method" not in kwargs:
            raise KeyError("Require a method definition")
        else:
            accepted_methods=["DCVGamma"]
            required_args={"DCVGamma":["area"]}
            if kwargs["method"] not in accepted_methods:
                raise KeyError("{0} method not implemented - only {1}".format(kwargs["method"], accepted_methods))
            for key in required_args[kwargs["method"]]:
                if key not in kwargs:
                    raise KeyError("{0} method requires {1} parameter".format(kwargs["method"], key))
        if isinstance(file, list):
            raise NotImplementedError
        elif isinstance(file, str):
            counter=0
            failiure=True
            while failiure==True and counter<5:
                try:
                    data=np.loadtxt(file, skiprows=counter)
                    failiure=False
                except:
                    counter+=1
            if failiure==True:
                data=np.loadtxt(file,skiprows=5)
            time=data[:,0]
            current=data[:,2]
            potential=data[:,1]
            dcpeak=sci._Heuristics.DCV_peak_area(time, potential, current, kwargs["area"])
            dcpeak.draw_background_subtract()
            plt.show()