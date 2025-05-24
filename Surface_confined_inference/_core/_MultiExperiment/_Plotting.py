import Surface_confined_inference as sci
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
class PlotManager:
    def __init__(self, composed_class):
        self._cls=composed_class
        self.grouping_keys=self._cls.grouping_keys
        self.group_to_class=self._cls.group_to_class
        self.group_to_parameters=self._cls.group_to_parameters
        self.group_to_conditions=self._cls.group_to_conditions
        self.all_harmonics=self._cls._all_harmonics        
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
        current_len=0
        line_list=[]
        for i in range(0, len(data_list)):
            xaxis=range(current_len, current_len+len(data_list[i]))
            if kwargs["label_list"] is not None:
                label=kwargs["label_list"][i]
            else:
                label=None
            l1, =axis.plot(xaxis, data_list[i], label=label, alpha=kwargs["alpha"], linestyle=kwargs["linestyle"], color=kwargs["colour"], lw=kwargs["lw"], path_effects=kwargs["patheffects"])
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
                current_maximum = np.max(np.array([arrayed[x][m][i,:] for x in range(num_plots)]), axis=None)
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
    def results(self, input_list, **kwargs):
        if "pre_saved" not in kwargs:
            kwargs["pre_saved"]=False     
        if kwargs["pre_saved"]==False:
            if isinstance(input_list[0], list) is False:
                input_list=[input_list]
        else:
            if isinstance(input_list, dict) is True:
                input_list=[input_list]
        self.plot_results(self.process_simulation_dict(input_list, pre_saved=kwargs["pre_saved"]), **kwargs)  
    def process_simulation_dict(self, input_list, pre_saved):
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
    def plot_results(self, simulation_values, **kwargs):
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
        if "axes" not in kwargs:
            if len(self.grouping_keys)%2!=0:
                num_cols=(len(self.grouping_keys)+1)/2
                fig,axes=plt.subplots(2, int(num_cols))
                axes[1, -1].set_axis_off()
            else:
                num_cols=len(self.grouping_keys)/2
                fig,axes=plt.subplots(2, int(num_cols))
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
                all_data=[self._cls.scale(self._cls.classes[x]["data"], groupkey, x) for x in self.group_to_class[groupkey]]
                all_times=[self._cls.classes[x]["times"] for x in self.group_to_class[groupkey]]
                label_list=[",".join(x.split("-")[1:]) for x in self.group_to_class[groupkey]]
                all_simulations=[x[groupkey] for x in simulation_values]
                
                    
                
                if "type:ft" not in groupkey:
                    self.plot_stacked_time(ax, all_data, label_list=label_list)
                    for q in range(0, len(all_simulations)):
                        if kwargs["sim_plot_options"]=="simple":
                            axis, time_lines=self.plot_stacked_time(ax, all_simulations[q], alpha=0.75, linestyle=linestyles[q%4], colour="black")
                        else:
                            axis, time_lines=self.plot_stacked_time(ax, all_simulations[q], alpha=0.75, linestyle=linestyles[q%4], colour=plot_colours[q], patheffects=path_effects, lw=kwargs["sim_plot_options"]["lw"])
                        if kwargs["interactive_mode"]==True:
                            self.simulation_plots[groupkey]=time_lines
                else:
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
                    if kwargs["sim_plot_options"]=="simple":
                        colors = [None] + ["black"] * len(all_simulations)#
                        patheffects=[None]*(len(all_simulations)+1)
                    else:
                        colors=[None]
                        patheffects=[None]
                        for r in range(0, len(plot_colours)):
                            colors+=[plot_colours[r]]
                            patheffects+=[path_effects]
                    dmax=np.max([np.max(x, axis=None) for x in data_harmonics], )        
                    scaled_harmonics=self.process_harmonics(plot_harmonics, additional_maximum=dmax)
                    axis, line_list=self.plot_scaled_harmonics(ax, scaled_harmonics,
                                                alpha=alphas, 
                                                linestyle=line_styles, 
                                                colour=colors, 
                                                label_list=label_list,
                                                scale=True,
                                                patheffects=patheffects)
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
        if kwargs["interactive_mode"]==False:
            if kwargs["savename"] is not None:
                fig.savefig(kwargs["savename"], dpi=500)
                plt.close()
            else:
                plt.show()