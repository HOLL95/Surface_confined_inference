import copy
import itertools
import re
from string import ascii_uppercase

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.spatial import ConvexHull

import Surface_confined_inference as sci


class PlotManager:
    def __init__(self, composed_class, results_array=None, pre_saved=False):
        self._cls=composed_class
        self.grouping_keys=self._cls.grouping_keys
        self.group_to_class=self._cls.group_to_class
        self.group_to_parameters=self._cls.group_to_parameters
        self.group_to_conditions=self._cls.group_to_conditions
        self.all_harmonics=self._cls._all_harmonics 
        self.class_keys=self._cls.class_keys
        self.results_array=results_array
        self.pre_saved=pre_saved
        self._all_parameters=self._cls._all_parameters
        
    def add_legend(self, ax, groupkey, target_cols=3):
        num_labels=len(self.group_to_class[groupkey])
        if num_labels<target_cols:
            num_cols=num_labels
            height=1
        else:
            num_cols=target_cols
            if num_labels%target_cols!=0:
                extra=1
            else:
                extra=0
            height=int(num_labels//target_cols)+extra
        ylim=ax.get_ylim()
        
        ax.set_ylim([ylim[0]-(abs(max(ylim)))*0.1*height, ylim[1]])
        ax.legend(loc="lower center", bbox_to_anchor=[0.5, 0], ncols=num_cols, handlelength=1)
                

    def plot_stacked_time(self, axis, data_list, **kwargs):
        from scipy.signal import envelope
        if "colour" not in kwargs:
            kwargs["colour"]=None
        if "linestyle" not in kwargs:
            kwargs["linestyle"]="-"
        if "alpha" not in kwargs:
            kwargs["alpha"]=1
        if "label_list" not in kwargs:
            kwargs["label_list"]=None
        if "patheffects" not in kwargs:
            kwargs["patheffects"]=None
        if "lw" not in kwargs:
            kwargs["lw"]=None
        if "envelope" not in kwargs:
            kwargs["envelope"]=False
        
        current_len=0
        line_list=[]
        for i in range(0, len(data_list)):
            xaxis=range(current_len, current_len+len(data_list[i]))
            if kwargs["label_list"] is not None:
                label=kwargs["label_list"][i]
            else:
                label=None
            if kwargs["envelope"]==False:
                l1, =axis.plot(xaxis, data_list[i], label=label, alpha=kwargs["alpha"], linestyle=kwargs["linestyle"], 
                                                                                        color=kwargs["colour"], 
                                                                                        lw=kwargs["lw"], 
                                                                                        path_effects=kwargs["patheffects"])
            else:
                y1, _=envelope(data_list[i], (10, None))
                y2, _=envelope(data_list[i]*-1, (10, None))
                l_1, =axis.plot(xaxis, y1, label=label, alpha=kwargs["alpha"], linestyle=kwargs["linestyle"], 
                                                                                        color=kwargs["colour"], 
                                                                                        lw=kwargs["lw"], 
                                                                                        path_effects=kwargs["patheffects"])
                l2, =axis.plot(xaxis, y2*-1, label=label, alpha=kwargs["alpha"], linestyle=kwargs["linestyle"], 
                                                                                        color=l_1.get_color(), 
                                                                                        lw=kwargs["lw"], 
                                                                                        path_effects=kwargs["patheffects"])
                l1=[l_1, l2]                                                                       
                
            current_len+=len(data_list[i])
            line_list.append(l1)
        axis.set_xticks([])
        
        return axis, line_list
    def process_harmonics(self, harmonics_list, **kwargs):
        """
        Process harmonics data and calculate all appropriate scalings.
        
        Parameters:
        -----------
        harmonics_list : list
            List of harmonics data to process
        **kwargs : dict
            Additional keyword arguments:
            - harmonics: list of harmonics to process (default: None)
            - scale: whether to scale harmonics (default: True)
            - residual: whether to compute residuals (default: False)
            
        Returns:
        --------
        dict
            Dictionary containing processed data and parameters for plotting
        """
        # Set default values if not provided
        if "harmonics" not in kwargs:
            kwargs["harmonics"] = self.all_harmonics
        if "scale" not in kwargs:
            kwargs["scale"] = True
        if "residual" not in kwargs:
            kwargs["residual"] = False
        if "additional_maximum" not in kwargs:
            kwargs["additional_maximum"]=0
            
        # Process the harmonics list
        arrayed = harmonics_list
        
        
        maximum = max(np.max(np.max([[np.max(y, axis=None) for y in x] for x in arrayed])), kwargs["additional_maximum"])

        #print(maximum)
        #print(np.max(arrayed, axis=None),kwargs["additional_maximum"], "*")
        # Get dimensions
        num_plots = len(arrayed)
        num_experiments = len(arrayed[0])
        num_harmonics = len(kwargs["harmonics"]) if kwargs["harmonics"] is not None else (len(arrayed[0][0]))
        
        # Validate residual option
        if kwargs["residual"] == True and num_plots != 2:
            raise ValueError("Can only do two sets of harmonics for a residual plot")
        
        # Calculate scaling factors and offsets
        scaled_data = []
        for m in range(num_experiments):
            exp_data = []
            for i in range(num_harmonics):
                harmonic_data = []
                # Calculate maximum for current harmonic across all plots
                temp_currents=[arrayed[x][m][i,:] for x in range(num_plots)]
                current_maximum = np.max(np.array([max(x) for x in temp_currents]), axis=None)
                # Calculate offset for stacking
                offset = (num_harmonics - i) * 1.1 * maximum
                # Calculate scaling ratio
                ratio = maximum / current_maximum if kwargs["scale"] else 1
                
                for j in range(num_plots):
                    xdata = range(len(arrayed[j][m][i,:]))
                    ydata = ratio * arrayed[j][m][i,:] + offset
                    harmonic_data.append((xdata, ydata))
                
                exp_data.append(harmonic_data)
            scaled_data.append(exp_data)
        
        # Prepare result dictionary
        result = {
            "scaled_data": scaled_data,
            "dimensions": {
                "num_plots": num_plots,
                "num_experiments": num_experiments,
                "num_harmonics": num_harmonics
            },
            "maximum": maximum
        }
        #scaled data is returned as experiment -> harmonic -> plot
        return result
    def _check_results_loaded(self):
        if hasattr(self._cls, "_results_array") is True and hasattr(self._cls, "_best_results") is True:
            return True
        else:
            return False
    def plot_scaled_harmonics(self, axis, processed_data, **kwargs):
        """
        Plot scaled harmonics data.
        
        Parameters:
        -----------
        axis : matplotlib.axes.Axes
            Axis to plot on
        processed_data : dict
            Dictionary containing processed data from process_harmonics
        **kwargs : dict
            Additional keyword arguments for styling:
            - colour: list of colors for each plot
            - linestyle: list of line styles
            - lw: list of line widths
            - patheffects: list of path effects
            - alpha: list of alpha values
            - label_list: list of labels for each experiment
            - utils_colours: list of colors to use for different experiments
            
        Returns:
        --------
        tuple
            (axis, line_list) - the axis and list of plotted lines
        """
        scaled_data = processed_data["scaled_data"]
        dimensions = processed_data["dimensions"]
        num_plots = dimensions["num_plots"]
        num_experiments = dimensions["num_experiments"]
        num_harmonics = dimensions["num_harmonics"]
        
        # Process styling parameters
        # Set default values
        if "colour" not in kwargs:
            kwargs["colour"] = [None]
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = ["-"]
        if "lw" not in kwargs:
            kwargs["lw"] = [None]
        if "patheffects" not in kwargs:
            kwargs["patheffects"] = [None]
        if "alpha" not in kwargs:
            kwargs["alpha"] = [1]
        if "label_list" not in kwargs:
            kwargs["label_list"] = None
            
        # Validate styling parameters
        style_keys = ["colour", "linestyle", "alpha", "lw", "patheffects"]
        for key in style_keys:
            if isinstance(kwargs[key], list) is False:
                raise ValueError(f"{key} needs to be wrapped into a list")
            
            if len(kwargs[key]) != num_plots:
                if len(kwargs[key]) == 1:
                    kwargs[key] = [kwargs[key][0] for _ in range(num_plots)]
                else:
                    raise ValueError(f"{key} needs to be the same length as the number of plots")
        
        # Prepare for plotting
        current_len = 0
        line_list = []
        
        # Get default colors
        utils_colours = kwargs.get("utils_colours", None)
        
        # Plot each experiment
        for m in range(num_experiments):
            
            
            # Plot each harmonic
            for i in range(num_harmonics):
                current_line_list = []
                # Plot each dataset
                for j in range(num_plots):
                    xdata, ydata = scaled_data[m][i][j]
                    
                    # Adjust x-axis for continuous plotting
                    xaxis = [x + current_len for x in xdata]
                    
                    # Set label for the first line of each experiment
                    if i == 0 and j == 0 and kwargs["label_list"] is not None:
                        label = kwargs["label_list"][m]
                    else:
                        label = None
                    
                    # Set color
                    if isinstance(kwargs["colour"][j], list) or isinstance(kwargs["colour"][j], np.ndarray):
                        colour = kwargs["colour"][j]
                    elif kwargs["colour"][j] is None:
                        # Use utils_colours if provided, otherwise use default
                        colour = utils_colours[m] if utils_colours is not None else f"C{m}"
                    else:
                        colour = kwargs["colour"][j]
                    
                    # Plot the line
                    l1, = axis.plot(
                        xaxis, 
                        ydata, 
                        label=label, 
                        alpha=kwargs["alpha"][j], 
                        linestyle=kwargs["linestyle"][j], 
                        color=colour, 
                        lw=kwargs["lw"][j], 
                        path_effects=kwargs["patheffects"][j]
                    )
                    
                    current_line_list.append(l1)
                line_list.append(current_line_list)
            # Update current length for the next harmonics
            if len(scaled_data[m][i][0][0]) > 0:
                current_len += len(scaled_data[m][i][0][0])
            
                
        
        # Remove x-axis ticks
        axis.set_xticks([])
        
        return axis, line_list
    def results(self,  **kwargs):
        possible_args=set(["parameters", "pareto_index", "best"])
        kwargset=set(kwargs.keys())
        intersect=possible_args-kwargset
        if len(intersect)!=2:
            raise ValueError(f"Only one of ({possible_args}) allowed as arg - found {intersect}")
        else:
            arg=list(kwargset.intersection(possible_args))[0]
        total_all_simulations=[]
        
        if arg=="parameters":
            param_values=np.array(kwargs[arg])
            simulate_all=True
        else:
            if self._check_results_loaded() is True:
                if arg in ["pareto_index", "best"]:
                    if arg=="best" and kwargs[arg]=="all":
                        kwargs[arg]=list(range(0, len(self.grouping_keys)))
                    elif isinstance(kwargs[arg], int):
                        kwargs[arg]=[kwargs[arg]]
                    simulate_all=True
                    for i in range(0, len(kwargs[arg])):
                        element=self._cls._results_array[kwargs[arg][i]]
                        if "saved_simulation" in element:
                            simulate_all=False
                        else:
                            if simulate_all==False:
                                raise ValueError("Saved simulation not found")
                    if simulate_all==True:
                        param_values=[]
                        for i in range(0, len(kwargs[arg])):
                            element=self._cls._results_array[kwargs[arg][i]]
                            param_values.append([element["parameters"][x] for x in self._cls._all_parameters])
                        param_values=np.array(param_values)
            else:
                raise ValueError(f"For {arg} argument results have to be loaded through `BaseMultiExperiment.results_loader()`")
        if simulate_all==True:
            kwargs["deced"]=False
            dims=param_values.shape
            if len(dims)==1:
                param_values=np.array([param_values])
                dims=param_values.shape
            for i in range(0, dims[0]):
                values=param_values[i,:]
                if all([x>=0 and x<=1 for x in values]) != self._cls._internal_options.normalise: 
                    print(f"Warning - are values correctly normalised? Loaded class has normalised set to {self._cls._internal_options.normalise}")
                #print("here")
                vals=self._cls.evaluate(values)
                total_all_simulations.append(vals)
                
                
        elif simulate_all==False:
            kwargs["deced"]=True
            for i in range(0, len(kwargs[arg])):
                    simulations={}
                    element=self._cls._results_array[kwargs[arg][i]]
                    for classkey in self._cls.class_keys:
                        vals=np.loadtxt(element["saved_simulation"][classkey]["address"])
                        simulations[classkey]=vals[:,element["saved_simulation"][classkey]["col"]]
                    total_all_simulations.append(simulations)
        scaled_simulations=[]
        for j in range(0, len(total_all_simulations)):
            groupsims={}
            for i in range(0, len(self.grouping_keys)):
                groupkey=self.grouping_keys[i]
                groupsims[groupkey]=[self._cls.scale(copy.deepcopy(total_all_simulations[j][x]), groupkey, x) for x in self.group_to_class[groupkey]]
            scaled_simulations.append(groupsims)

        #case1 range of parameters
        #case2 index to pareto
        #case3 just "best"
        #sequential or not>>>>
        if "sequential" not in kwargs:
            kwargs["sequential"]=False
        if "savename" not in kwargs:
            kwargs["savename"]=None
        if kwargs["sequential"]==True:
            for i in range(0, len(scaled_simulations)):
                if kwargs["savename"] is not None:
                    if kwargs["savename"].split(".")[-1]=="png":
                        raise ValueError("Dont put .png at the end of savename ({0})".format(kwargs["savename"]))
                    if arg=="best":
                        savename=kwargs["savename"]+"_"+self.grouping_keys[i]
                    else:
                        savename=kwargs["savename"]+"_index_"+str(i)
                else:
                    savename=None

                oldsavename=kwargs["savename"]
                kwargs["savename"]=savename
                if arg=="best":
                    kwargs["target_key"]=[self.grouping_keys[i]]
                if "best" in kwargs:
                    kwargs["target_key"]=self.grouping_keys[kwargs["best"][i]]
                self.plot_results([scaled_simulations[i]], **kwargs)  
                kwargs["savename"]=oldsavename

        else:
            if "best" in kwargs:
                    kwargs["target_key"]=[self.grouping_keys[x] for x in kwargs["best"]]
            self.plot_results(scaled_simulations, **kwargs)
    def process_simulation_dict(self, input_list):
        total_all_simulations=[]
        for p in input_list:
            if pre_saved==False:
                simulation_vals=self._cls.evaluate(p)
            else:
                simulation_vals=p
            groupsims={}
            for i in range(0, len(self.grouping_keys)):
                groupkey=self.grouping_keys[i]
                groupsims[groupkey]=[self._cls.scale(simulation_vals[x], groupkey, x) for x in self.group_to_class[groupkey]]
        total_all_simulations.append(groupsims)
        return total_all_simulations
   
    def _configure_plot_axis(self):
        if len(self.grouping_keys)%2!=0:
            num_cols=(len(self.grouping_keys)+1)/2
            fig,axes=plt.subplots(2, int(num_cols))
            axes[1, -1].set_axis_off()
        else:
            num_cols=len(self.grouping_keys)/2
            fig,axes=plt.subplots(2, int(num_cols))
        return fig,axes
    def master_harmonics_plotter(self, all_data, all_simulations, all_times, ax, 
                              label_list, plot_colours, linestyles, path_effects, 
                              **kwargs):
        """
        Process harmonics for data and simulations, then plot them.
        
        Parameters:
        -----------
        all_data : list
            List of experimental data arrays
        all_simulations : list
            List of simulation data arrays
        all_times : list
            List of time arrays corresponding to data/simulations
        ax : matplotlib axis
            Axis object for plotting
        label_list : list
            List of labels for the plots
        plot_colours : list
            List of colors for plotting
        linestyles : list
            List of line styles
        path_effects : list
            Path effects for styling
        **kwargs : dict
            Additional keyword arguments (must contain "sim_plot_options")
        
        Returns:
        --------
        axis : matplotlib axis
            The axis object with plotted harmonics
        line_list : list
            List of line objects from the plot
        """
        
        num_experiments = len(all_data)
        num_simulations = len(all_simulations) + 1  # +1 for the data
        
        # Initialize harmonics_list with proper dimensions (m×n×o)
        # m = num_simulations, n = num_experiments, o = length of harmonics
        harmonics_list = []
        
        # First, calculate harmonics for the actual data
        data_harmonics = [
            np.array(np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics)))
            for t, i in zip(all_times, all_data)
        ]
        harmonics_list.append(data_harmonics)
        
        # Then calculate harmonics for each simulation
        for q in range(0, len(all_simulations)):
            sim = all_simulations[q]
            sim_harmonics = [
                np.array(np.abs(sci.plot.generate_harmonics(t, i, hanning=True, one_sided=True, harmonics=self.all_harmonics)))
                for t, i in zip(all_times, sim)
            ]
            harmonics_list.append(sim_harmonics)
        
        plot_harmonics = harmonics_list
        
        # Create lists for styling parameters
        alphas = [1] + [0.75] * len(all_simulations)
        line_styles = ["-"] + [linestyles[l % len(linestyles)] for l in range(0, len(all_simulations))]
        
        if kwargs["sim_plot_options"] == "simple":
            colors = [None] + ["black"] * len(all_simulations)
            patheffects = [None] * (len(all_simulations) + 1)
        else:
            if len(plot_colours)!=(num_simulations):
                colors = [None]
                patheffects = [None]
                for r in range(0, len(plot_colours)):
                    colors += [plot_colours[r]]
                    patheffects += [path_effects]
            else:
                colors=plot_colours
                patheffects = [None] + [path_effects]*(num_simulations-1)
            
                
        
        dmax = np.max([np.max(x, axis=None) for x in data_harmonics])
        scaled_harmonics = self.process_harmonics(plot_harmonics, additional_maximum=dmax)
        
        axis, line_list = self.plot_scaled_harmonics(ax, scaled_harmonics,
                                                alpha=alphas,
                                                linestyle=line_styles,
                                                colour=colors,
                                                label_list=label_list,
                                                scale=True,
                                                patheffects=patheffects)
        
        return axis, line_list
    def plot_results(self, simulation_values, **kwargs):
        for classkey in self.class_keys:
            if "data" not in self._cls.classes[classkey]:
                raise ValueError(f"No data associated with {classkey}")
        linestyles=[ "dashed", "dashdot","dotted",]
        if "target_key" not in kwargs:
            target_key=[None]
        else:
            if isinstance(kwargs["target_key"], str):   
                target_key=[kwargs["target_key"]]
            elif isinstance(kwargs["target_key"], list):
                target_key=kwargs["target_key"]
        if "simulation_labels" not in kwargs:
            kwargs["simulation_labels"]=None
        elif len(kwargs["simulation_labels"]) != len(simulation_values):
            raise ValueError("simulation_labels ({0}) not same length as provided simulation_values ({1}) ".format(len(kwargs["simulation_labels"]), len(simulation_values)))
        if "savename" not in kwargs:
            kwargs["savename"]=None
        if "show_legend" not in kwargs:
            kwargs["show_legend"]=False
        if "envelope" not in kwargs:
            kwargs["envelope"]=False
        if "axes" not in kwargs or kwargs["axes"] is None:
            fig,axes=self._configure_plot_axis()
        else:
            axes=kwargs["axes"]
        self.plot_line_dict={}
        defaults={
                "cmap":mpl.colormaps['plasma'],
                "foreground":"black",
                "lw":0.4,
                "strokewidth":3,
                "colour_range":[0.2, 0.75]
                
            }
        if "sim_plot_options" not in kwargs:
            kwargs["sim_plot_options"]="simple"
        elif kwargs["sim_plot_options"]=="default" or kwargs["sim_plot_options"]=="simple":
            pass
        else:
            for key in defaults.keys():
                if key not in kwargs["sim_plot_options"]:
                    kwargs["sim_plot_options"]=defaults[key]
        if kwargs["sim_plot_options"]!="simple":
            plot_colours=kwargs["sim_plot_options"]["cmap"](
                np.linspace(kwargs["sim_plot_options"]["colour_range"][0],
                kwargs["sim_plot_options"]["colour_range"][1],
                len(simulation_values))
            )
            path_effects=[pe.Stroke(linewidth=kwargs["sim_plot_options"]["strokewidth"], foreground=kwargs["sim_plot_options"]["foreground"]), pe.Normal()]
        else:
            path_effects=None 
        if "interactive_mode" not in kwargs:
            kwargs["interactive_mode"]=False
        if kwargs["interactive_mode"]==True:
            if isinstance(simulation_values[0], dict) is False or len(simulation_values)>1:
                raise ValueError("In interactive mode, you can only submit a single dictioanry of simulation values")
            self.simulation_plots={"maxima":{}, "data_harmonics":{}}
        for i in range(0, len(self.grouping_keys)):
                
                groupkey=self.grouping_keys[i]
                ax=axes[i%2, i//2]
                
                if groupkey in target_key:
                    for axis in ['top','bottom','left','right']:
                        ax.spines[axis].set_linewidth(4)
                        ax.tick_params(width=4)
                ax.set_title(groupkey, fontsize=8)
                if kwargs["deced"]==True:
                    data_key="deced_"
                else:
                    data_key=""
                all_data=[self._cls.scale(self._cls.classes[x][data_key+"data"], groupkey, x) for x in self.group_to_class[groupkey]]
                all_times=[self._cls.classes[x][data_key+"times"] for x in self.group_to_class[groupkey]]
                label_list=[",".join(x.split("-")[1:]) for x in self.group_to_class[groupkey]]
                all_simulations=[x[groupkey] for x in simulation_values]
                
                    
                
                if "type:ft" not in groupkey:
                    
                    self.plot_stacked_time(ax, all_data, label_list=label_list)
                    for q in range(0, len(all_simulations)):
                        

                        if kwargs["sim_plot_options"]=="simple":
                            axis, time_lines=self.plot_stacked_time(ax, all_simulations[q], alpha=0.75, linestyle=linestyles[q%3], colour="black", envelope=kwargs["envelope"])
                        else:
                            axis, time_lines=self.plot_stacked_time(ax, all_simulations[q], alpha=0.75, linestyle=linestyles[q%3], colour=plot_colours[q], patheffects=path_effects, lw=kwargs["sim_plot_options"]["lw"], envelope=kwargs["envelope"])
                        if kwargs["interactive_mode"]==True:
                            self.simulation_plots[groupkey]=time_lines
                        
                
                else:
                    if kwargs["sim_plot_options"]=="simple":
                        plot_colours=None
                    
                    self.master_harmonics_plotter(
                        all_data, all_simulations, all_times, ax, 
                              label_list, plot_colours, linestyles, path_effects, 
                              **kwargs
                    )
                    if kwargs["interactive_mode"]==True:
                        self.simulation_plots[groupkey]=line_list#lines are stored columnwise per plot and rowise per harmonics
                        self.simulation_plots["maxima"][groupkey]=dmax
                        self.simulation_plots["data_harmonics"][groupkey]=data_harmonics
                if kwargs["show_legend"]==True:
                    self.add_legend(ax, groupkey)
                if kwargs["simulation_labels"] is not None:
                    
                    if i%2==0 and i//2==len(self.grouping_keys)//4:
                        twinx=ax.twinx()
                        twinx.set_yticks([])
                        ylim=ax.get_ylim()
                        xlim=ax.get_xlim()
                        for r in range(0, len(kwargs["simulation_labels"])):
                            if kwargs["sim_plot_options"]=="simple":
                                twinx.plot(xlim[0],ylim[0], color="black", linestyle=linestyles[r%4])
                            else:
                                twinx.plot(xlim[0],ylim[0], color=plot_colours[r], linestyle=linestyles[r%4], path_effects=path_effects, label=kwargs["simulation_labels"][r])
                        twinx.legend(ncols=len(kwargs["simulation_labels"]), bbox_to_anchor=[0.5, -0.1], loc="center")
        fig=plt.gcf()                                                                  
        fig.set_size_inches(16, 10)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        if "return_axes" not in kwargs:
            kwargs["return_axes"]=False
        if kwargs["return_axes"]==True:
            return fig, dict(zip(self.grouping_keys,[ axes[x//2, x%2] for x in range(0, len(self.grouping_keys))]))
        if kwargs["interactive_mode"]==False:
            if kwargs["savename"] is not None:
                if kwargs["savename"].split(".")[-1]!=".png":
                    kwargs["savename"]+=".png"
                fig.savefig(kwargs["savename"], dpi=500)
                plt.close()
            else:
                plt.show()
    def plot_2d_pareto(self,**kwargs):
        if "address" not in kwargs:
            address=None
        else:
            address=kwargs["address"]
        if "ax" not in kwargs:
            kwargs["ax"]=None
        else:
            axis=kwargs["ax"]
        if "size" not in kwargs:
            kwargs["size"]=50
        if "keylabel" not in kwargs:
            kwargs["keylabel"]=True
        if kwargs["keylabel"]==True:
            letterdict=dict(zip(self.grouping_keys, ascii_uppercase[:len(self.grouping_keys)]))
            legend_dict={key:f"{letterdict[key]} : {key}" for key in self.grouping_keys}
            if "keylabel_legend" not in kwargs:
                kwargs["keylabel_legend"]=True
        if "envelope_threshold" not in kwargs:
            kwargs["envelope_threshold"]=0.1
        if "show_envelope" not in kwargs:
            kwargs["show_envelope"]=False
        
        if address is None:
            if self._check_results_loaded() is False:
                raise ValueError("No `address` argument provided and results not loaded through directory")
            else:
                if kwargs["ax"] is None:
                    fig,axis=plt.subplots(len(self.grouping_keys), len(self.grouping_keys))
                for j in range(0, len(self.grouping_keys)):
                    keyy=self.grouping_keys[j]
                   
                    for k in range(0, len(self.grouping_keys)):
                        ax=axis[j,k]
                        keyx=self.grouping_keys[k]
                        xscores=[x["scores"][keyx] for x in self._cls._results_array]
                        yscores=[y["scores"][keyy] for y in self._cls._results_array]

                        if j<k:
                            ax.set_axis_off()
                        
                        if j==k:
                            ax.hist(xscores, density=True)
                            
                                
                        if j>k:
                            ax.scatter(xscores, yscores)
                            if kwargs["show_envelope"]==True:
                                idx=self._get_2d_neighours(xscores, yscores, threshold=kwargs["envelope_threshold"])
                                xplot=[xscores[x] for x in idx]
                                yplot=[yscores[x] for x in idx]
                                ax.scatter(xplot, yplot, c="red")
                num_metrics=len(self.grouping_keys)
                for i in range(0, num_metrics):
                    for j in range(0, num_metrics):
                      
                        if i==num_metrics-1:
                            if kwargs["keylabel"]==True:
                                xlabel=letterdict[self.grouping_keys[j]]
                            else:
                                xlabel=self.grouping_keys[j]
                            axis[i,j].set_xlabel(xlabel)
                        if j==0:
                            if kwargs["keylabel"]==True:
                                ylabel=letterdict[self.grouping_keys[i]]
                            else:
                                ylabel=self.grouping_keys[i]
                            if i==0:
                                ax.set_ylabel("Count")
                            else:
                                axis[i,j].set_ylabel(ylabel)
                if kwargs["keylabel"]==True:
                    for i in range(0, num_metrics):
                        axis[0,-1].plot(0, 0, label=legend_dict[self.grouping_keys[i]])
                if kwargs["keylabel_legend"]==True:
                    axis[0,-1].legend(loc="center", handlelength=0)
                return axis
        else:
            raise NotImplementedError("Manual `address` loading currently not implemented")
    def _un_normalise_parameters(self,):
        if self._check_results_loaded() is False:
            raise ValueError("Need to load results through `sci.BaseMultiExperiment.resuls_loader()`")
        return_array=np.zeros((len(self._cls._results_array),len(self._cls._all_parameters),))
        for i in range(0, len(self._cls._all_parameters)):
            norm_param=self._cls._all_parameters[i]
            if "_offset" in norm_param:
                assign=[y["parameters"][norm_param] for y in self._cls._results_array]
            else:
                if norm_param not in self._cls.boundaries.keys():
                    norm_param="_".join(self._cls._all_parameters[i].split("_")[:-1])
                assign=[sci._utils.un_normalise(y["parameters"][self._cls._all_parameters[i]], self._cls.boundaries[norm_param]) for y in self._cls._results_array]
            return_array[:,i]=assign
        return return_array
    def _get_2d_neighours(self,xscores, yscores, **kwargs):
            from sklearn.neighbors import BallTree
            points = np.column_stack([xscores, yscores])
            hull = ConvexHull(points).vertices
            hull_pairs=np.array([points[x,:] for x in hull])
            sorted_x=np.argsort(hull_pairs[:,0])
            sorted_y=np.argsort(hull_pairs[:,1])
            terminal_x=np.where(hull_pairs[:,0]==hull_pairs[sorted_x[0],0])[0][0]
            current_y=hull_pairs[sorted_x[0],[1]]
            end=False
            lower_envelope=[hull_pairs[terminal_x,:]]
            for i in range(terminal_x, len(sorted_x)):
                if hull_pairs[i,1]*0.9>current_y:
                    end=True
                if end==True:
                    break
                lower_envelope.append(hull_pairs[i,:])
                current_y=hull_pairs[i,1]
            if end==False:
                for i in range(0, len(sorted_x)):
                    if hull_pairs[i,1]*0.9>current_y:
                        end=True
                    if end==True:
                        break
                    lower_envelope.append(hull_pairs[i,:])
                    current_y=hull_pairs[i,1]
            lower_envelope=np.array(lower_envelope)
            new_array = [tuple(row) for row in lower_envelope]
            lower_envelope = np.unique(new_array, axis=0)
            
            #ax.plot(lower_envelope[:,0], lower_envelope[:,1], color="green")
            
            xrange=np.linspace(lower_envelope[0,0], lower_envelope[-1, 0], 1000)
            yrange=np.interp(xrange, lower_envelope[:,0], lower_envelope[:,1])
            boundary=np.column_stack((xrange, yrange))
            points=np.column_stack((xscores, yscores))
            tree = BallTree(points)
            std_devs = np.std(points, axis=0)
            absolute_threshold = kwargs["threshold"] * np.mean(std_devs)
            indices_list = tree.query_radius(boundary, r=absolute_threshold).ravel()
            empty_array=[]
            for elem in indices_list:
                empty_array+=list(elem)
            indices_list=np.unique(empty_array)
            return indices_list
    def depths_from_dict(self, infodict, score_array):
        from depth.model.DepthEucl import DepthEucl
        combinations=list(itertools.combinations(self._all_parameters,2))
        joined_keys=[f"{x[0]}&{x[1]}" for x in combinations]
        combinations_dict={x:{} for x in joined_keys}
        setlist=[]
        for m in range(0, len(combinations)):
            informative=[]
            for cparam in combinations[m]:
                informative.append([x for x in range(0, len(self.grouping_keys)) if self.grouping_keys[x] in infodict[cparam]])
            informative_union=set(informative[0]).union(set(informative[1]))
            unique=True
            for existing_set in setlist:
                if existing_set==informative_union:
                    unique=False
                    break
            if unique==True:
                setlist.append(informative_union)
            combo_key=f"{combinations[m][0]}&{combinations[m][1]}"
            combinations_dict[combo_key]["setid"]=informative_union
        depth_lists=[]
        for m in range(0, len(setlist)):
            if len(setlist[m])<2:
                depth_lists.append(None)
            else:
                new_score_array=np.column_stack([score_array[:,x] for x in setlist[m]])
                model=DepthEucl().load_dataset(new_score_array)
                depthDataset=model.spatial(evaluate_dataset=True)
                depth_lists.append(depthDataset)
        return depth_lists, combinations_dict, setlist
    def pareto_parameter_plot(self, **kwargs):
       
        if self._check_results_loaded() is False:
            raise ValueError("No `address` argument provided and results not loaded through directory")
        if "axes" not in kwargs:
            fig,ax=plt.subplots(len(self._all_parameters), len(self._all_parameters))
        else:
            ax=kwargs["axes"]
        if "single_colour" not in kwargs:
            kwargs["single_colour"]=None
        if "true_values" not in kwargs:
            kwargs["true_values"]=None
        if "show_depths" not in kwargs:
            kwargs["show_depths"]=False
        if kwargs["show_depths"]==False:
            kwargs["order_by_depth"]=False
        else:
            from depth.model.DepthEucl import DepthEucl
            if "order_by_depth" not in kwargs:
                kwargs["order_by_depth"]=True
            if kwargs["single_colour"] is not None:
                raise ValueError("If show_depths is True, single_colour must be `None` not {0}".format(kwargs["single_colour"]))
        if "edgecolour" not in kwargs:
            kwargs["edgecolour"]=None
        if "size" not in kwargs:
            kwargs["size"]=1
        if "target_keys" not in kwargs:
            kwargs["target_keys"]=None
        if kwargs["target_keys"] is not None and kwargs["show_depths"] is not False:
            raise ValueError("Only one of `target_keys` and `show_depths` can be active at a time")
        if kwargs["target_keys"] is not None or kwargs["show_depths"] is not False:
            colorbar=True
        else:
            colorbar=False
        if kwargs["target_keys"] is not None:
            if len(kwargs["target_keys"])!=2:
                raise ValueError("`target_keys` argument must be of length 2!, not {}".format(len(kwargs["target_keys"])))
            elif all([key in self.grouping_keys for key in kwargs["target_keys"]]) is False:
                raise ValueError("`target_keys` need to be from `grouping_keys`")
        if "information_score" not in kwargs:
            kwargs["information_score"]=None

        if kwargs["information_score"] is not None:
            if "show_non_info" not in kwargs:
                kwargs["show_non_info"]=True 
            if "common_scale" not in kwargs:
                kwargs["common_scale"]=True
        if "cmap" not in kwargs:
            kwargs["cmap"]=mpl.colormaps["seismic"]
        all_params=self._all_parameters
        bounds=self._cls.boundaries
        groups=self.grouping_keys
        full_param_set=np.array([[y["parameters"][x] for x in all_params] for y in self._cls._results_array])
        de_normalised_array=self._un_normalise_parameters()
        #full_score_set=np.array([[y["scores"][x] for x in groups] for y in self._cls._results_array])
        if colorbar==True:
            colorax=[0.6, ax[0,0].get_position().bounds[1], 0.3, ax[0,0].get_position().bounds[3]]
            colour_ax=fig.add_axes(colorax)
            colour_ax.set_yticks([])
            
        if kwargs["target_keys"] is not None:
            xscores=[y["scores"][kwargs["target_keys"][0]] for y in self._cls._results_array]
            yscores=[y["scores"][kwargs["target_keys"][1]] for y in self._cls._results_array]
            score_array=np.zeros((len(xscores), 2))
            scores=np.column_stack((xscores, yscores))
           
            for i in range(0, 2):
                score_array[:,i]=np.min(scores[:,i])/scores[:,i]
            colour_array=score_array[:,0]/score_array[:,1]
            
            table_width=0.15
            
            left_table_ax=fig.add_axes([colorax[0]-table_width/2, colorax[1]-0.03, table_width, 0.01])
            right_table_ax=fig.add_axes([colorax[0]+colorax[2]-table_width/2, colorax[1]-0.03, table_width, 0.01])
            xlim=colour_ax.get_xlim()
            colour_ax.set_xticks([xlim[0], np.mean(xlim), xlim[1]], ["Better fit to:", "Equally good fit", "Better fit to:"])
            colour_ax.imshow(np.vstack((np.linspace(0,1,256),np.linspace(0,1,256))), aspect='auto', cmap=kwargs["cmap"])
            minc=min(colour_array)
            maxc=max(colour_array)
            labeldict={"experiment":"Experiment", "type":"Domain"}
            labelval={"Experiment":lambda x:x, "Domain":lambda x, returndict={"ts":"Time", "ft":"Frequency"}:returndict.get(x)}
            unitdict={"Hz":"Frequency (Hz)", "mV":"Amplitude (mV)", "V":"Amplitude (V)"}
            equalitydict={"lesser":"< ", "equals":"", "geq":r"$\geq$"}
            pattern = r"\d+(?:\.\d+)?"
            tableaxes=[left_table_ax, right_table_ax]
            for i in range(0, len(kwargs["target_keys"])):
                table=[]
                split1=kwargs["target_keys"][i].split("-")
                split2=[x.split(":") for x in split1]
                for j in range(0, len(split2)):
                    if split2[j][0] in labeldict:
                        arg1=labeldict[split2[j][0]]
                        units=False
                    else:
                        units=True
                        key=[key for key in unitdict.keys() if key in split2[j][1]][0]
                        arg1=unitdict[key]
                    if units==False:
                        arg2=labelval[arg1](split2[j][1])
                    else:
                        
                        match = re.search(pattern, split2[j][1])
                        arg2=f"{equalitydict[split2[j][0]]}{match.group(0)}"
                    table.append([arg1, arg2])
                tab=tableaxes[1-i].table(cellText=table, cellLoc="left")
                tab.auto_set_font_size(False)
                tab.set_fontsize(10)
                #tab.scale(1.5, 1.5)
                tableaxes[1-i].set_axis_off()
        elif kwargs["show_depths"] is not False:
            colour_ax.set_xlabel("Statistical depth")
            score_array=np.array([[x["scores"][y] for y in self.grouping_keys] for x in self._cls._results_array])
            if kwargs["information_score"] is not None:
                depth_lists, combinations_dict, setlist=self.depths_from_dict(kwargs["information_score"], score_array)
                combinations=list(itertools.combinations(self._all_parameters,2))
                joined_keys=list(combinations_dict.keys())
            model=DepthEucl().load_dataset(score_array)
            depthDataset=model.spatial(evaluate_dataset=True)
            
            xidx=np.argsort(depthDataset)
            #params=self._un_normalise_parameters()
            minval=min(depthDataset)
            maxval=max(depthDataset)
            if kwargs["information_score"] is not None:
                
                idxs=[["" for x in all_params] for y in all_params]
                minval=2
                maxval=0
                alphaspan=np.linspace(0.2, 1, len(self.grouping_keys)+1)

                for i in range(0, len(all_params)):
                    for j in range(0, len(all_params)):
                        if i>j:     
                            xparam=all_params[j]
                            yparam=all_params[i]
                            for m in range(0, len(combinations)):
                                splitkey=combinations[m]
                                joinkey=joined_keys[m]
                                if xparam in splitkey and yparam in splitkey:
                                    setid=combinations_dict[joinkey]["setid"]
                                    setidx=[x for x in range(0, len(setlist)) if setlist[x]==setid][0]
                                    depthDataset=depth_lists[setidx]
                                    if depthDataset is not None:
                                        minval=min(minval, min(depthDataset))
                                        maxval=max(maxval, max(depthDataset))
                                    else:
                                        depthDataset="lightslategray"
                                    
                                
                                
                            
                            log_dict={
                                #"coloured_alphas":plotalphas,
                                #"coloured":[[params[x,j] for x in score_idx], [params[x,i] for x in score_idx]],
                                "colours":depthDataset,
                                #"black":[[params[y,j] for y in range(0, len(params)) if y not in score_idx], [params[y,i] for y in range(0, len(params)) if y not in score_idx]]
                            }
                            idxs[i][j]=log_dict
                num_points=len(self.grouping_keys)
            norm=plt.Normalize(vmin=minval, vmax=maxval)                 
            plt.colorbar(mappable=cm.ScalarMappable(
                norm=norm,
                cmap=kwargs["cmap"]), 
                cax=colour_ax,
                orientation="horizontal")
            colour_ax.set_xlabel("Spatial depth")
            if kwargs["information_score"] is not None and kwargs["common_scale"]==False:
                colour_ax.set_xticks([minval, maxval])
                colour_ax.set_xticklabels(["min", "max"])
            #colour_ax.imshow(np.vstack((np.linspace(0,1,256),np.linspace(0,1,256))), aspect='auto', cmap=kwargs["cmap"])
            colour_array=depthDataset
            
      
        
        for i in range(0, len(all_params)):
            for j in range(0, len(all_params)):
                if i>=j:
                    plot_axis=[]
                    
                    for x in [j,i]:
                       
                        plot_axis.append(de_normalised_array[:,x])
                        
                if i>j:  
                    xparam=all_params[j]
                    yparam=all_params[i]
                    if kwargs["true_values"] is not None:
                        
                       
                        if xparam in kwargs["true_values"]:
                            ax[i,j].axvline(kwargs["true_values"][xparam], color="black", linestyle="--")
                        if yparam in kwargs["true_values"]:
                            ax[i,j].axhline(kwargs["true_values"][yparam], color="black", linestyle="--")
                        if xparam not in kwargs["true_values"]:
                            for key in kwargs["true_values"].keys():
                                norm_param="_".join(key.split("_")[:-1])
                                if norm_param==xparam:
                                    ax[i,j].axvline(kwargs["true_values"][key], color="black", linestyle="--")
                        if yparam not in kwargs["true_values"]:
                            for key in kwargs["true_values"].keys():
                                norm_param="_".join(key.split("_")[:-1])
                                if norm_param==yparam:
                                    ax[i,j].axhline(kwargs["true_values"][key], color="black", linestyle="--")
                    if kwargs["show_depths"] is not False:
                        if kwargs["information_score"] is not None:
                            colour=idxs[i][j]["colours"]
                        else:
                            colour=depthDataset
                    else:
                        colour=kwargs["single_colour"]
                    if kwargs["order_by_depth"] ==True:
                        if isinstance(colour, str) is False:
                            cidx=np.argsort(colour)
                            plot_axis[0]=[plot_axis[0][x] for x in cidx]
                            plot_axis[1]=[plot_axis[1][x] for x in cidx]
                            colour=sorted(colour)
                    if kwargs["information_score"] is not None:
                        if kwargs["common_scale"] is False:
                            norm_arg=None
                        else:
                            norm_arg=norm
                        ax[i,j].scatter(plot_axis[0], plot_axis[1], 
                            kwargs["size"], cmap=kwargs["cmap"], norm=norm_arg, 
                            c=colour,
                            edgecolor=kwargs["edgecolour"],
                            )
                            
                    else:
                        if colorbar==True:
                            plot_args=dict(c=colour, cmap=kwargs["cmap"], alpha=1,edgecolor=kwargs["edgecolour"],)
                        else:
                            plot_args=dict(alpha=1,edgecolor=kwargs["edgecolour"],c=colour)
                        ax[i,j].scatter(plot_axis[0], plot_axis[1], kwargs["size"], **plot_args)#
                    if j==0:
                        ax[i,j].set_ylabel(all_params[i])
                        if abs(np.mean(plot_axis[1]))<1e-2:
                            ax[i,j].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
                    else:
                        ax[i,j].set_yticks([])
                
                elif i==j:  
                    xaxis=plot_axis[0]
                    n, bin_edges, patches=ax[i,j].hist(xaxis, edgecolor=kwargs["edgecolour"],bins=25, density=True)
                    if j!=0:
                        ax[i,j].set_yticks([])
                    #hist,bin_edges = np.histogram(xaxis, 25)
                    if kwargs["target_keys"] is not None:
                        colours=np.zeros(len(n))
                        for m in range(1, len(bin_edges)):
                            location=np.where((xaxis>bin_edges[m-1]) & (xaxis<bin_edges[m]))
                            colours[m-1]=np.mean(colour_array[location])
                    
                            colours=[kwargs["cmap"](sci._utils.normalise(x, [minc, maxc])) for x in colours]
                            for c, p in zip(colours, patches):
                                plt.setp(p, 'facecolor', c)
                    if i+j!=0:
                        ax[i,j].set_yticks([])
                    

                else:
                    ax[i,j].set_axis_off()
                
                if i==len(all_params)-1:
                    ax[i,j].set_xlabel(all_params[j])
                    ax[i,j].tick_params(axis='x', labelrotation=35)
                    if abs(np.mean(plot_axis[0]))<1e-2:
                        ax[i,j].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
                else:
                    ax[i,j].set_xticks([])
                    
        ax[0,0].set_ylabel("Density")
        fig=plt.gcf()
        fig.set_size_inches(16, 11)
        plt.subplots_adjust(top=0.985,
        bottom=0.088,
        left=0.053,
        right=0.992,
        hspace=0.307,
        wspace=0.202)
        if "savename" not in kwargs:
            kwargs["savename"]=None
        if kwargs["savename"] is not None:
            fig.savefig(kwargs["savename"], dpi=500)
        return fig,ax
    def show_pareto_range(self,**kwargs):
        if self._check_results_loaded() is False:
            raise ValueError("Need to load results through `sci.BaseMultiExperiment.results_loader()`")
        if "envelope" not in kwargs:
            kwargs["envelope"]=False
        fig,axes=self._configure_plot_axis()
        load_dict={key:[] for key in self.class_keys }
        all_scores=np.array([[x["scores"][y] for y in self.grouping_keys] for x in self._cls._results_array]) #column is the score, row is the entry
        scored_argsorts=np.argsort(all_scores, axis=0)

        results=self._cls._results_array
        #fig, axes=self.plot_results([], return_axes=True)
        for i in range(0, len(self.class_keys)):
            classkey=self.class_keys[i]
            for z in [0,-1]:#first is best, second is worst
                idx=scored_argsorts[z,i]
                vals=np.loadtxt(results[idx]["saved_simulation"][classkey]["address"])
                load_dict[classkey].append(vals[:,results[idx]["saved_simulation"][classkey]["col"]])
        plot_dict={key:{} for key in self.grouping_keys}
        for i in range(0, len(self.grouping_keys)):
            gkey=self.grouping_keys[i]
            for j in range(0, len(self.group_to_class[gkey])):
                ckey=self.group_to_class[gkey][j]
                if ckey not in plot_dict[gkey]:
                    plot_dict[gkey][ckey]=[]
                for m in range(0, len(load_dict[ckey])):
                    plot_dict[gkey][ckey].append(self._cls.scale(copy.deepcopy(load_dict[ckey][m]), gkey, ckey))

        for i in range(0, len(self.grouping_keys)):
            gkey=self.grouping_keys[i]
            ax=axes[i%2, i//2]
            simulations=[
                    [plot_dict[gkey][ckey][x] for ckey in self.group_to_class[gkey]]
                    for x in range(0, 2)
                ]
            data=[self._cls.scale(copy.deepcopy(self._cls.classes[ckey]["data"]), gkey, ckey) for ckey in self.group_to_class[gkey]]
            
            plotting=simulations
           
            if "type:ft" not in gkey:
                
                colours=["#0099ff","red"]
                means=[np.mean([np.mean(np.abs(y)) for y in x]) for x in plotting]

                idx=np.flip(np.argsort(means))
                
                for q in range(0, len(idx)):
                    
                    self.plot_stacked_time(ax, plotting[idx[q]], colour=colours[idx[q]], envelope=kwargs["envelope"])
                self.plot_stacked_time(ax, data, colour="black", alpha=0.5,envelope=kwargs["envelope"])
            else:
                colours=["red", "#0099ff"]
                all_times=[self._cls.classes[x]["times"] for x in self.group_to_class[gkey]]
                self.master_harmonics_plotter(
                        data, simulations, all_times, ax, 
                              None, ["black"]+colours, "-", None,sim_plot_options="complex" 
                    )
        return fig,axes

            
                    
                    
                            
                            
