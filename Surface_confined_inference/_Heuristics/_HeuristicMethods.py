
import matplotlib.pyplot as plt
import numpy as np

import Surface_confined_inference as sci


class HeuristicMethod:
    def __init__(self,file, **kwargs):
        if "method" not in kwargs:
            raise KeyError("Require a method definition")
        else:
            accepted_methods=["DCVGamma", "DCVTrumpet"]
            required_args={"DCVGamma":["area"], "DCVTrumpet":["area", "data_loc", "save_address", "fitting", "header_length"]}
            if kwargs["method"] not in accepted_methods:
                raise KeyError("{0} method not implemented - only {1}".format(kwargs["method"], accepted_methods))
            for key in required_args[kwargs["method"]]:
                if key not in kwargs:
                    raise KeyError("{0} method requires {1} parameter".format(kwargs["method"], key))
        if kwargs["method"]=="DCVGamma":
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
        elif kwargs["method"]=="DCVTrumpet":
            
            if isinstance(file, list) is False:
                raise ValueError("Need to pass a list of filenmaes for trumpet k0 determination")
            sci._Heuristics.Automated_trumpet(file_list=file, trumpet_file_name=kwargs["save_address"],data_loc=kwargs["data_loc"], area=kwargs["area"], skiprows=kwargs["header_length"])