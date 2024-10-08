import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import os
import re
import copy
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from decimal import Decimal
import Surface_confined_inference as sci
import pints
class DCV_peak_area():
    
    def __init__(self, times, potential, current, area, **kwargs):
        
        self.times=times
        self.area=area
        self.current=current
        self.potential=potential
        self.func_switch={"1":self.poly_1, "2":self.poly_2, "3":self.poly_3, "4":self.poly_4}
        self.all_plots=["total", "subtract", "bg", "poly"]
        self.check_status=dict(zip(self.all_plots, [False]*len(self.all_plots)))
        self.vlines=[]
        self.scatter_points=[]
        self.peak_positions=[None, None]
        self.show_peaks="Lines"
        if "func_order" not in kwargs:
            func_order=2
        else:
            func_order=kwargs["func_order"]
        if "position_list" not in kwargs:
            warnings.warn("No peak position list found")
            self.position_list={}
        elif isinstance(kwargs["position_list"], dict) is False:
            raise ValueError("Position list needs to be a dictionary (defined by {})")
        else:
            self.position_list=kwargs["position_list"]
        if "data_filename" not in kwargs:
            self.data_filename=None
        else:
            self.data_filename=kwargs["data_filename"]
        
        if "scan_rate" not in kwargs:
            self.scan_rate=None
        else:
            self.scan_rate=kwargs["scan_rate"]
        self.func_order=func_order
        self.middle_idx=list(potential).index(max(potential))
    def poly_1(self, x, a, b):
        return a*x+b
    def poly_2(self, x, a, b, c):
        return (a*x**2)+b*x+c
    def poly_3(self, x, a, b, c, d):
        return (a*x**3)+(b*x**2)+c*x+d
    def poly_4(self, x, a, b, c, d, e):
        return (a*x**4)+(b*x**3)+(c*x**2)+d*x+e
    def background_subtract(self, params):
        #1Lower intersting section sweep1
        #2Upper interesting section sweep1
        #3Lower intersting section sweep2
        #4Upper interesting section sweep2
        #5 start of first sweep
        #6end of first sweep
        #7 start of second sweep
        #8end of second sweep
        func=self.func_switch[self.func_order]
        half_len=len(self.potential[self.middle_idx:])
        first_idx=np.where(self.potential>params[4])[0][0]
        second_idx=np.where(self.potential[:self.middle_idx]>params[5])[0][0]
        #fig, ax=plt.subplots()
        #ax.plot(self.potential, self.current)
        #ax.plot(self.potential[np.where(self.potential[:self.middle_idx]>params[5])], self.current[np.where(self.potential[:self.middle_idx]>params[5])])
        #plt.show()
        fourth_idx=half_len+np.where(self.potential[self.middle_idx:]<params[6])[0][0]
        third_idx=half_len+np.where(self.potential[self.middle_idx:]<params[7])[0][0]
        #print(len(self.potential), params[6], params[7])
        #print(first_idx, second_idx, third_idx, fourth_idx)
        idx_1=[first_idx, third_idx]
        idx_2=[second_idx, fourth_idx]
        interesting_section=[[params[0], params[1]], [params[2], params[3]]]
        current_results=self.current
        subtract_current=np.zeros(len(current_results))
        fitted_curves=np.zeros(len(current_results))
        nondim_v=self.potential
        time_results=self.times
        #plt.plot(self.potential, self.current)
        return_arg={}
        for i in range(0, 2):
            current_half=current_results[idx_1[i]:idx_2[i]]
            time_half=time_results[idx_1[i]:idx_2[i]]
         
            
            
            volt_half=self.potential[idx_1[i]:idx_2[i]]
            #plt.plot(volt_half, current_half)
            #print(volt_half)
            #print(interesting_section[i][0], interesting_section[i][1])
            noise_idx=np.where((volt_half<interesting_section[i][0]) | (volt_half>interesting_section[i][1]))
            signal_idx=np.where((volt_half>interesting_section[i][0]) & (volt_half<interesting_section[i][1]))
            noise_voltages=volt_half[noise_idx]
            noise_current=current_half[noise_idx]
            noise_times=time_half[noise_idx]
            #plt.plot(noise_voltages, noise_current)
            #plt.show()
            popt, pcov = curve_fit(func, noise_times, noise_current)
            fitted_curve=[func(t, *popt) for t in time_half]
            return_arg["poly_{0}".format(i)]=[volt_half, fitted_curve]
            
            #plt.plot(volt_half, fitted_curve, color="red")
            sub_c=np.subtract(current_half, fitted_curve)
            subtract_current[idx_1[i]:idx_2[i]]=sub_c
            fitted_curves[idx_1[i]:idx_2[i]]=fitted_curve
            #plt.plot(volt_half[signal_idx], sub_c[signal_idx])
            area=simpson(sub_c[signal_idx], x=time_half[signal_idx])
            return_arg["bg_{0}".format(i)]=[noise_voltages, noise_current]
            return_arg["subtract_{0}".format(i)]=[volt_half[signal_idx], sub_c[signal_idx]]
            gamma=abs(area/(self.area*96485.3321))
            return_arg["gamma_{0}".format(i)]="{:.3E}".format(Decimal(gamma))
        return return_arg
    def get_slider_vals(self,):
        params=[self.slider_array[key].val for key in self.slider_array.keys()] 
        if params[0]<params[4]:
            params[0]=params[4]
        if params[1]>params[5]:
            params[1]=params[5]
        if params[2]<params[6]:
            params[2]=params[6]
        if params[3]>params[7]:
            params[3]=params[7]
        return params
    def update(self, value):
        #1Lower intersting section sweep1
        #2Upper interesting section sweep1
        #3Lower intersting section sweep2
        #4Upper interesting section sweep2
        #5 start of first sweep
        #6end of first sweep
        #7 start of second sweep
        #8end of second sweep

        params=self.get_slider_vals()
        get_vals=self.background_subtract(params)
        txt=["f", "b"]
        plot_dict=dict(zip(self.all_plots[1:], [self.subtracted_lines, self.bg_lines,self.red_lines,]))
        
        if self.check_status["total"]==True:
            self.total_line.set_data(0,0)
        else:
            self.total_line.set_data(self.potential, self.current)
        for elem in self.scatter_points:
            elem.remove()
        for elem in self.vlines:
            elem.remove()
        self.vlines=[]
        self.scatter_points=[]
        for i in range(0,2):
            
            for key in plot_dict.keys():
                if self.check_status[key]==True:
                     plot_dict[key][i].set_data(0,0)
                else:
                    plot_dict[key][i].set_data(get_vals["{0}_{1}".format(key, i)][0],get_vals["{0}_{1}".format(key, i)][1])
                if key=="subtract":
                    current=get_vals["{0}_{1}".format(key, i)][1]
                    potential=get_vals["{0}_{1}".format(key, i)][0]
                    abs_current=abs(current)
                    current_max=max(abs_current)
                    loc=np.where(abs_current==current_max)
                    potential_max=potential[loc][0]
                    actual_current_max=current[loc]
                    self.peak_positions[i]=potential_max
                    if self.show_peak_position!="Hide":
                       
                        
                        if self.show_peaks=="Points":

                            self.scatter_points.append(self.all_Ax.scatter(potential_max, actual_current_max, s=100,color="purple", marker="x"))
                        elif self.show_peaks=="Lines":
                            self.vlines.append(self.all_Ax.axvline(potential_max, color="purple", linestyle="--"))
            self.gamma_text[i].set_text("$\\Gamma_"+txt[i]+"="+get_vals["gamma_{0}".format(i)]+"$ mol cm$^{-2}$")
            
            self.all_Ax.relim()
            self.all_Ax.autoscale_view()
    def draw_background_subtract(self,**kwargs):
        
        fig, ax=plt.subplots()
        self.all_Ax=ax
        self.total_line, = ax.plot(self.potential, self.current, lw=2)

        self.red_lines=[ax.plot(0,0, color="red")[0] for x in range(0, 2)]
        self.subtracted_lines=[ax.plot(0,0, color="black", linestyle="--")[0] for x in range(0, 2)]
        self.bg_lines=[ax.plot(0,0, color="orange")[0] for x in range(0, 2)]
        fig.subplots_adjust(left=0.1, bottom=0.35, right=0.605)
        params=["Ox start", "Ox end", "Red start", "Red end", "Forward start", "Forward end", "Reverse start", "Reverse end"]
        init_e0=(min(self.potential)+max(self.potential))/2
        init_e0_pc=0.1
        init_start=init_e0-init_e0_pc
        init_end=init_e0+init_e0_pc
        val=0.05
        if "init_vals" not in kwargs:
            kwargs["init_vals"]=None
        if kwargs["init_vals"]==None:
            init_param_values=[init_start, init_end, init_start, init_end, min(self.potential)+val, max(self.potential)-val, min(self.potential)+val, max(self.potential)-val]
        else:
            init_param_values=kwargs["init_vals"]
      
        param_dict=dict(zip(params, init_param_values))
        interval=0.25/8
        titleax = plt.axes([0.65, 0.84, 0.1, 0.04])
        titleax.set_title("BG subtraction")
        titleax.set_axis_off()
        resetax = plt.axes([0.65, 0.82, 0.1, 0.04])
        text_ax= plt.axes([0.61, 0.4, 0.1, 0.04])
        self.button = Button(resetax, 'Reset',  hovercolor='0.975')
        self.button.on_clicked(self.reset)
        polyax = plt.axes([0.65, 0.625, 0.1, 0.15])
        self.radio2 = RadioButtons(
        polyax, ('2', '3', '4'),
        )
        #polyax = plt.axes([0.65, 0.475, 0.1, 0.15])
        #self.radio1 = RadioButtons(
        #polyax, ("Show BG", "Hide BG"),
        #)
        self.show_bg=True
        self.func_order="2"
        txt=["f", "b"]
        get_vals=self.background_subtract(init_param_values)
        self.gamma_text=[text_ax.text(0.0, 1-(i*1), "$\\Gamma_"+txt[i]+"="+get_vals["gamma_{0}".format(i)]+"$ mol cm$^{-2}$") for i in range(0, 2)]
        text_ax.set_axis_off()
        

        kinetic_title_ax= plt.axes([0.85, 0.84, 0.1, 0.04])
        kinetic_title_ax.set_title("Peak position")
        kinetic_title_ax.set_axis_off()
        #save_Ax = plt.axes([0.84, 0.82, 0.12, 0.04])
        #self.button = Button(save_Ax, 'Save position',  hovercolor='0.975')

        axbox = plt.axes([0.84, 0.82, 0.12, 0.04])
        axbox.text(0.22, 1.2,"Scan rate", transform=axbox.transAxes)
        self.text_box = TextBox(axbox, "", textalignment="center")
        if self.data_filename is not None:
            self.all_Ax.set_title(self.data_filename)
        if self.scan_rate is not None:
            self.text_box.set_val(self.scan_rate)
        elif self.data_filename is not None:
            
            lower_name=self.data_filename.lower()
            
            if "mv" in lower_name:
               
                match=re.findall(r"\d+(?:\.\d+)?(?=mv)", lower_name)
                if len(match)==1:
                    self.text_box.set_val(match[0])
        hideax= plt.axes([0.85, 0.625, 0.12, 0.15])
        self.check = CheckButtons(
            ax=hideax,
            labels=("Hide total", "Hide sub", "Hide BG", "Hide poly"),
            

        )
        polyax = plt.axes([0.85, 0.475, 0.12, 0.15])
        self.radio3 = RadioButtons(
        polyax, ("Lines", "Points","Hide"),
        )
        self.save_text=axbox.text(0, -0.6, "Not saved", transform=axbox.transAxes)

        self.text_box.on_submit(self.submit_scanrate)
        self.radio2.on_clicked(self.radio_button)
        self.check.on_clicked(self.hiding)
        #self.radio1.on_clicked(self.show_bg_func)
        self.radio3.on_clicked(self.show_peak_position)
        for element in [self.radio2, self.check, self.radio3]:
            element.on_clicked(self.update)
        for radios in [self.radio2]:
            print(vars(radios))
            #for circle in radios.circles:
            #    circle.set_radius(0.09)
        self.slider_array={}
        class_array={}
        for i in range(0, len(params)):
            axfreq = fig.add_axes([0.2, 0.25-(i*interval), 0.65, 0.03])
            self.slider_array[params[i]] = Slider(
                ax=axfreq,
                label=params[i],
                valmin=min(self.potential),
                valmax=max(self.potential),
                valinit=init_param_values[i],
            )
            self.slider_array[params[i]].on_changed(self.update)
        self.update(init_param_values)
        fig.set_size_inches(9.5, 8)
    def reset(self, event):
        [self.slider_array[key].reset() for key in self.slider_array.keys()] 
    def radio_button(self, value):
        self.func_order=value
    def hiding(self, label):
        self.check_status=dict(zip(self.all_plots, self.check.get_status()))
    def show_peak_position(self, value):
        self.show_peaks=value
    def submit_scanrate(self,expression):
        
        try:
            key=float(expression)
            self.position_list[key]=self.peak_positions
            self.save_text.set_text("{0}mV added".format(key))
        except:
            self.save_text.set_text("Not a scan rate")
    def get_scale_dict(self):
        return self.position_list

class Automated_trumpet(DCV_peak_area):
    def __init__(self, file_list, trumpet_file_name, **kwargs):
        if "filetype" not in kwargs:
            kwargs["filetype"]="Ivium"
        if kwargs["filetype"]!="Ivium":
            raise ValueError("Other files not supported")
        if "skiprows" not in kwargs:
            kwargs["skiprows"]=0
        if "data_loc" not in kwargs:
            kwargs["data_loc"]=""
        if "area" not in kwargs:
            raise ValueError("Need to define an area")
        if "data_order" not in kwargs:
            kwargs["data_order"]=["time", "potential", "current"]
        update_file_list, scan_rates=self.sort_file_list(file_list)
        if isinstance(update_file_list, bool) is False:
            file_list=update_file_list
        scale_dict={}
        pot_idx=kwargs["data_order"].index("potential")
        curr_idx=kwargs["data_order"].index("current")

        for i in range(0,len(file_list)):
            DCV_data=np.loadtxt(os.path.join(kwargs["data_loc"],file_list[i]), skiprows=kwargs["skiprows"])
            if DCV_data.shape[1]<3:
                warnings.warn("Attempting to get scan rate from potential - life is easier if the files have a time column")
                if isinstance(scan_rates, bool) is True:
                    raise ValueError("Extracting time from potential only works if you know the scan rate")
                
                potential=DCV_data[:,pot_idx]
                current=DCV_data[:,curr_idx]
                diffed=abs(np.diff(potential))
                if np.std(diffed)>1e-6:
                    raise ValueError("Either potential is in the wrong place or potential too noisy to recover time")
                current_scan_rate=scan_rates[i]/1000
                timesteps=np.divide(diffed, current_scan_rate)
                time=np.append(0,np.cumsum(timesteps))
            else:
                time_idx=kwargs["data_order"].index("time")
                time=DCV_data[:,time_idx]
                current=DCV_data[:,curr_idx]
                potential=DCV_data[:,pot_idx]


            if isinstance(scan_rates, bool) is False:
                scan_arg=scan_rates[i]
            else:
                scan_arg=None
            if i==0:
                init_vals=None
            super().__init__(time,potential, current, kwargs["area"], data_filename=file_list[i], position_list=scale_dict, scan_rate=scan_arg)
            self.draw_background_subtract(init_vals=init_vals)
            plt.show()
            
            scale_dict=self.get_scale_dict()
            init_vals=self.get_slider_vals()
        key_list=sorted([int(x) for x in scale_dict.keys()])
        trumpet_file=open(trumpet_file_name, "w")
        for key in key_list:
            
            line=(",").join([str(key), str(scale_dict[key][0]), str(scale_dict[key][1])])+"\n"
            trumpet_file.write(line)
        trumpet_file.close()
    def sort_file_list(self, file_list):
        element_list=[]
        scan_list=[]
        split_list_list=[]
        for i in range(0, len(file_list)):
            filename=file_list[i].lower()
            mv_has_scan=True
            if "mv" in filename:
                try:
                    match=re.findall(r"\d+(?:\.\d+)?(?=(?:_|\s)?mv)", filename)[0]
                    scan_list.append(float(match))
                except:
                    mv_has_scan=False
            elif "mv" not in filename or mv_has_scan==False: 
                split_list=re.split(r"[\s.;_-]+", filename)
                split_list_list.append(split_list)
                new_list=[]
                for element in split_list:
                    try:
                        new_list.append(float(element))
                    except:
                        continue
                element_list.append(new_list)
        if len(scan_list)!=len(file_list):
            for i in range(0, len(element_list[0])):
                column=[element_list[x][i] for x in range(0, len(element_list))]
                if len(column)!=len(set(column)):
                    continue
                else:
                    maximum=max(column)
                    minimum=min(column)
                    if np.log10(maximum-minimum)<2:
                        continue
                    else:
                        scan_list+=column
        if len(scan_list)!=len(file_list):
            
            print("Have not been able to automatically sort the files. If you want this to work, either add `mv` after the scan rate in the filename, or have a consistent naming scheme")
            return False,False
        else:
            sorted_idx=np.argsort(scan_list)
            return_list=[file_list[x] for x in sorted_idx]
            return return_list, sorted(scan_list)
class TrumpetSimulator:
    def __init__(self,experiment_parameters, **kwargs):

        self.experiment_parameters=experiment_parameters
        self._scan_rates=[]
        
        self.volts={}
        self.times={}
        self.nd_times={}
        self.classes={}
        if "fixed_parameters" not in kwargs:
            self.fixed_parameters={"gamma":experiment_parameters["Surface_coverage"]}
        else:
            self.fixed_parameters=kwargs["fixed_parameters"]
        if "optim_list" not in kwargs:
            self.optim_list=[]
        else:
            self.optim_list=kwargs["optim_list"]
        if "sampling_factor" not in kwargs:
            self.sampling_factor=50
        else:
            self.sampling_factor=kwargs["sampling_factor"]
        if "scan_rates" in kwargs:
            self.scan_rates_defined=True
            self.scan_rates=kwargs["scan_rates"]
        else:
            self.scan_rates_defined=False
    @property
    def scan_rates(self):
        return self._scan_rates
    @scan_rates.setter
    def scan_rates(self, scan_rates,):
       
        for i in range(0, len(scan_rates)):
            sim_params=copy.deepcopy(self.experiment_parameters)
            fl_scan_rate=float(scan_rates[i])
            sim_params["v"]=fl_scan_rate
            self.classes[fl_scan_rate]=sci.SingleExperiment("DCV", sim_params)
           
            self.times[fl_scan_rate]=self.classes[fl_scan_rate].calculate_times(input_parameters=sim_params,dimensional=True, sampling_factor=self.sampling_factor)
            self.volts[fl_scan_rate]=self.classes[fl_scan_rate].get_voltage(self.times[fl_scan_rate], input_parameters=sim_params, dimensional=True)
            self.nd_times[fl_scan_rate]=self.classes[fl_scan_rate].nondim_t(self.times[fl_scan_rate])
            if self.optim_list!=[]:
                self.classes[fl_scan_rate].fixed_parameters=self.fixed_parameters
                self.classes[fl_scan_rate].optim_list=self.optim_list
        self.scan_rates_defined=True
        
        self._scan_rates=scan_rates
    @property 
    def optim_list(self):
        return self._optim_list
    @optim_list.setter
    def optim_list(self, parameters):
        if parameters!=[]:
            for key in self.classes.keys():
               
                self.classes[key].optim_list=parameters
        self._optim_list=parameters
        
    @property
    def fixed_parameters(self):
        return self._fixed_parameters
    @fixed_parameters.setter
    def fixed_parameters(self, fixed_parameters):
        for key in self.classes.keys():
                self.classes[key].fixed_parameters=fixed_parameters
        self._fixed_parameters=fixed_parameters

    def trumpet_positions(self, current, voltage):
        volts=np.array(voltage)
        amps=np.array(current)

        """if self.simulation_options["find_in_range"]!=False:
            dim_volts=self.e_nondim(volts)
            search_idx=tuple(np.where((dim_volts>self.simulation_options["find_in_range"][0])&(dim_volts<self.simulation_options["find_in_range"][1])))
            #fig, ax=plt.subplots(1,1)
            #ax.plot(dim_volts, amps)
            volts=volts[search_idx]
            amps=amps[search_idx]"""
        max_pos=volts[np.where(amps==max(amps))]
        min_pos=volts[np.where(amps==min(amps))]
        return max_pos, min_pos
       
    def n_outputs(self):
        return 2
    def n_parameters(self):
        return len(self._optim_list)
    def trumpet_plot(self, scan_rates, trumpets, **kwargs):
        if "ax" not in kwargs:
            ax=None
        else:
            ax=kwargs["ax"]
        if "label" not in kwargs:
            label=None
        else:
            label=kwargs["label"]
        if "description" not in kwargs:
            kwargs["description"]=False
        else:
            label=None
        if "log" not in kwargs:
            kwargs["log"]=np.log10
        if len(trumpets.shape)!=2:
            raise ValueError("For plotting reductive and oxidative sweeps together")
        else:
            colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
            if ax==None:
                plt.scatter(kwargs["log"](scan_rates),  trumpets[:, 0], label="Forward sim", color=colors[0])
                plt.scatter(kwargs["log"](scan_rates),  trumpets[:, 1], label="Reverse sim", color=colors[0], facecolors='none')
                plt.legend()
                plt.show()
            else:
                if kwargs["description"]==False:
                    if ax.collections:
                        pass
                    else:
                        self.color_counter=0
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 0], label=label, color=colors[self.color_counter])
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 1], color=colors[self.color_counter], facecolors='none')
                    self.color_counter+=1
                else:
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 0], label="$E_{p(ox)}-E^0$", color=colors[0])
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 1], label="$E_{p(red)}-E^0$", color=colors[0], facecolors='none')
    def TrumpetSimulate(self, parameters, scan_rates):
        for key in self.classes.keys():
            self.classes[key].optim_list=self._optim_list
            self.classes[key].fixed_parameters=self._fixed_parameters
        return self.simulate(parameters, scan_rates)    
    def simulate(self, parameters, scan_rates):
        if self.scan_rates_defined==False:
            self.scan_rates=scan_rates
        param_dict=dict(zip(self._optim_list, parameters))
        forward_sweep_pos=np.zeros(len(scan_rates))
        reverse_sweep_pos=np.zeros(len(scan_rates))
        for i in range(0, len(scan_rates)):
            #print(self.classes[scan_rates[i]]._internal_memory["input_parameters"]["v"], scan_rates[i])
            current=self.classes[scan_rates[i]].simulate(parameters, self.nd_times[scan_rates[i]])
            try:
                forward_sweep_pos[i], reverse_sweep_pos[i]=self.trumpet_positions(current, self.volts[float(scan_rates[i])])
            except:
                 forward_sweep_pos[i]=-1000
                 reverse_sweep_pos[i]=-1000
            #plt.plot(self.nd_times[scan_rates[i]], current)
        #plt.show()

        if "dcv_sep" in self._optim_list:
            forward_sweep_pos+=param_dict["dcv_sep"]
            reverse_sweep_pos-=param_dict["dcv_sep"]
        return np.column_stack((forward_sweep_pos, reverse_sweep_pos))
    def fit(self, scan_rates, positions,**kwargs):
        if "paralell" in kwargs:
            raise KeyError("You've mis-spelled paraLLeL")
        if "tolerance" not in kwargs:
            kwargs["tolerance"]=1e-6
        if "method" not in kwargs:
            kwargs["method"]=pints.CMAES
        if "runs" not in kwargs:
            kwargs["runs"]=1
        if "unchanged_iterations" not in kwargs:
            kwargs["unchanged_iterations"]=200
        if "parallel" not in kwargs:
            kwargs["parallel"]=True
        if "sigma0" not in kwargs:
            kwargs["sigma0"]=0.075
        if "starting_point" not in kwargs:
            kwargs["starting_point"]="random"
        if "boundaries" not in kwargs:
            raise KeyError("Requires boundaries for {0}".format(self._optim_list))
        if "plot_results" not in kwargs:
            kwargs["plot_results"]=False
        
        self.scan_rates=scan_rates
        for key in self.classes.keys():
            self.classes[key].boundaries=kwargs["boundaries"]
            self.classes[key].normalise_parameters=True
        problem=pints.MultiOutputProblem(self, scan_rates, positions)
        error=pints.SumOfSquaresError(problem)
        boundaries=pints.RectangularBoundaries(
                                            np.zeros(self.n_parameters()), 
                                            np.ones(self.n_parameters())
                                        )
        if kwargs["starting_point"]=="random":
            x0=np.random.random(self.n_parameters())
        else:
            between_0_and_1=[(x>0 and x<1) for x in kwargs["starting_point"]]
            if all(between_0_and_1)==True:

                x0=kwargs["starting_point"]
            else:
                x0=self.classes[key].change_normalisation_group(kwargs["starting_point"], "norm")
           
                
        if kwargs["sigma0"] is not list:
            sigma0=[kwargs["sigma0"] for x in range(0,self.n_parameters())]
        else:
            sigma0=kwargs["sigma0"]
        best_score=-1e20
        scores=np.zeros(kwargs["runs"])
        parameters=np.zeros((kwargs["runs"], self.n_parameters()))

        noises=np.zeros(kwargs["runs"])
        for i in range(0, kwargs["runs"]):
            optimiser=pints.OptimisationController(error, x0, sigma0=sigma0, boundaries=boundaries, method=kwargs["method"])
            optimiser.set_max_unchanged_iterations(iterations=kwargs["unchanged_iterations"], threshold=kwargs["tolerance"])
            optimiser.set_parallel(kwargs["parallel"])
            xbest, fbest=optimiser.run()
            scores[i]=fbest
            dim_params=list(self.classes[key].change_normalisation_group(xbest, "un_norm"))
            parameters[i,:]=dim_params
        sorted_idx=np.argsort(scores)
        best_param=parameters[sorted_idx[0],:]   
        print(xbest)
        print(list(best_param))
        for key in self.classes.keys():
            self.classes[key].normalise_parameters=False
            print(self.classes[key].fixed_parameters)
        if kwargs["plot_results"]==True:
            fig,ax=plt.subplots()
            self.trumpet_plot(scan_rates, positions, ax=ax, label="Data") 
            values=self.simulate(best_param, scan_rates)
            self.trumpet_plot(scan_rates, values, ax=ax, label="Simulation") 
            plt.show()
        return best_param