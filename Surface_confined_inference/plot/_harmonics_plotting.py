import copy

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import Surface_confined_inference as sci


def generate_harmonics(times, data, **kwargs):
    if "func" not in kwargs or kwargs["func"] == None:
        func = lambda x: x
    else:
        func = kwargs["func"]
    if "zero_shift" not in kwargs:
        kwargs["zero_shift"] = False
    if "hanning" not in kwargs:
        kwargs["hanning"] = False
    if "return_amps" not in kwargs:
        kwargs["return_amps"] = False

    if "return_fourier" not in kwargs:
        kwargs["return_fourier"] = False
    L = len(data)
    if kwargs["hanning"] == True:
        window = np.hanning(L)
        time_series = np.multiply(data, window)
    else:
        time_series = data
    if "one_sided" not in kwargs:
        kwargs["one_sided"] = True
    if "harmonics" not in kwargs:
        kwargs["harmonics"] = sci.maximum_availiable_harmonics(times, data)
        #kwargs["harmonics"]=list(range(1, 10))
    if "save_csv" not in kwargs:
        kwargs["save_csv"]=False
    
    elif kwargs["save_csv"] is not False and isinstance(kwargs["save_csv"], str)==False:
        raise TypeError("save_csv argument needs to be str, not {0}".format(type(kwargs["save_csv"])))
    num_harmonics = len(kwargs["harmonics"])
    if kwargs["return_amps"] == True:
        amps = np.zeros(num_harmonics)
    if "filter_val" not in kwargs:
        kwargs["filter_val"] = 0.5
    f = np.fft.fftfreq(len(time_series), times[1] - times[0])
    Y = np.fft.fft(time_series)
    input_frequency = sci.get_frequency(times, time_series)
    last_harm = kwargs["harmonics"][-1] * input_frequency
    frequencies = f
    harmonics = np.zeros((num_harmonics, len(time_series)), dtype="complex")
    if kwargs["return_fourier"] == True:
        one_side_idx = np.where((f > 0) & (f < (last_harm + (0.5 * input_frequency))))
        one_side_frequencies = f[one_side_idx]
        ft_peak_return = np.zeros((num_harmonics, len(f)), dtype="complex")
    
    for i in range(0, num_harmonics):
        true_harm = kwargs["harmonics"][i] * input_frequency
        # plt.axvline(true_harm, color="black")
        freq_idx = np.where(
            (frequencies < (true_harm + (input_frequency * kwargs["filter_val"])))
            & (frequencies > true_harm - (input_frequency * kwargs["filter_val"]))
        )
        # filter_bit=(top_hat[freq_idx])
        if kwargs["return_amps"] == True:

            abs_bit = abs(filter_bit)
            # print(np.real(filter_bit[np.where(abs_bit==max(abs_bit))]))
            amps[i] = np.real(filter_bit[np.where(abs_bit == max(abs_bit))])
        if kwargs["zero_shift"] == True:
            harmonics[i, 0 : len(filter_bit)] = func(filter_bit)
        else:
            f_domain_harmonic = np.zeros(len(Y), dtype="complex")
            for j in [-1, 1]:
                top_hat = copy.deepcopy(Y)  # (copy.deepcopy(Y[0:len(frequencies)]))
                top_hat[
                    np.where(
                        (
                            frequencies
                            > (j * true_harm + (input_frequency * kwargs["filter_val"]))
                        )
                        | (
                            frequencies
                            < true_harm * j - (input_frequency * kwargs["filter_val"])
                        )
                    )
                ] = 0
                f_domain_harmonic += top_hat

        if kwargs["return_fourier"] == False:
            if kwargs["one_sided"] == True:
                harmonics[i, :] = 2 * (np.fft.ifft(top_hat))

            else:
                harmonics[i, :] = np.fft.ifft(f_domain_harmonic)
        else:

            ft_peak_return[i, :] = f_domain_harmonic
    
    if kwargs["save_csv"] is not False:
        save_dict={"Time":times}
        for i in range(0, len(kwargs["harmonics"])):
            if kwargs["one_sided"]==True:
                save_dict["Harmonic {0}".format(kwargs["harmonics"][i])]=np.abs(harmonics[i, :])
            else:
                save_dict["Harmonic {0}".format(kwargs["harmonics"][i])]=np.real(harmonics[i, :])
        DataFrame(data=save_dict).to_csv(kwargs["save_csv"])
    if kwargs["return_amps"] == True:
        return harmonics, amps
    if kwargs["return_fourier"] == False:
        return harmonics
    else:
        return ft_peak_return, f


def single_oscillation_plot(times, data, **kwargs):
    if "colour" not in kwargs:
        kwargs["colour"] = None
    if "label" not in kwargs:
        kwargs["label"] = ""
    if "alpha" not in kwargs:
        kwargs["alpha"] = 1
    if "ax" not in kwargs:
        kwargs["ax"] = plt.subplots()
    if "start_time" not in kwargs:
        start_time = 3
    else:
        start_time = kwargs["start_time"]
    if "end_time" not in kwargs:
        end_time = int(times[-1] // 1) - 1
    else:
        end_time = kwargs["end_time"]
    if isinstance(end_time, int) and isinstance(start_time, int):
        step = 1
    else:
        if "oscillation_frequency" not in kwargs:
            raise ValueError("Need to define an oscillation_frequency")
        else:
            step = kwargs["oscillation_frequency"]
    full_range = np.arange(start_time, end_time, step)

    for i in range(0, len(full_range) - 1):
        data_plot = data[
            np.where((times >= full_range[i]) & (times < (i + full_range[i + 1])))
        ]
        time_plot = np.linspace(0, 1, len(data_plot))
        if i == start_time:
            line = kwargs["ax"].plot(
                time_plot,
                data_plot,
                color=kwargs["colour"],
                label=kwargs["label"],
                alpha=kwargs["alpha"],
            )
        else:
            line = kwargs["ax"].plot(
                time_plot, data_plot, color=kwargs["colour"], alpha=kwargs["alpha"]
            )
    return line


def inv_objective_fun(times, time_series, **kwargs):
    input_frequency = sci.get_frequency(times, time_series)
    if "func" not in kwargs:
        func = None
    else:
        func = kwargs["func"]
    if "dt" not in kwargs:
        dt = None
    else:
        dt = kwargs["dt"]
    if "harmonics" not in kwargs:
        kwargs["harmonics"] = sci.maximum_availiable_harmonics(times, time_series)
    func_not_right_length = False
    if func != None:
        likelihood = func(time_series)
        if len(likelihood) == (len(time_series) // 2) - 1:
            likelihood = np.append(likelihood, [0, np.flip(likelihood)])
        if len(likelihood) != len(time_series):
            func_not_right_length = True
    if func == None or func_not_right_length == True:
        if dt == None:
            raise ValueError("To create the likelihood you need to give a dt")
        L = len(time_series)
        window = np.hanning(L)
        # time_series=np.multiply(time_series, window)
        f = np.fft.fftfreq(len(time_series), dt)
        Y = np.fft.fft(time_series)
        top_hat = copy.deepcopy(Y)
        first_harm = (kwargs["harmonics"][0] * input_frequency) - (
            input_frequency * 0.5
        )
        last_harm = (kwargs["harmonics"][-1] * input_frequency) + (
            input_frequency * 0.5
        )
        print(first_harm, last_harm)
        abs_f = np.abs(f)
        Y[np.where((abs_f < (first_harm)) | (abs_f > last_harm))] = 0
        likelihood = Y
    time_domain = np.fft.ifft(likelihood)
    return time_domain


def plot_harmonics(**kwargs):
    label_list = []
    time_series_dict = {}
    harm_dict = {}
    if "hanning" not in kwargs:
        kwargs["hanning"] = False
    if "xaxis" not in kwargs:
        kwargs["xaxis"] = "time"
    if "plot_func" not in kwargs:
        kwargs["plot_func"] = np.real
    if "fft_func" not in kwargs:
        kwargs["fft_func"] = None
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = ""
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = ""
    if "legend" not in kwargs:
        kwargs["legend"] = {"loc": "center", "facecolor":"white", "framealpha":0.5}
    if "h_num" not in kwargs:
        kwargs["h_num"] = True
    if "one_sided" not in kwargs:
        kwargs["one_sided"] = True
    if "filter_val" not in kwargs:
        kwargs["filter_val"] = 0.5
    if "remove_xaxis" not in kwargs:
        kwargs["remove_xaxis"]=False
    if "ylabelpad" not in kwargs:
        kwargs["ylabelpad"]=4
    if "title" not in kwargs:
        kwargs["title"]=None
    if "nyticks" not in kwargs:
        kwargs["nyticks"]=None
    if "save_csv" not in kwargs:
        kwargs["save_csv"]=False
    if "clip_oscillations" not in kwargs:
        kwargs["clip_oscillations"]=None
    label_counter = 0
    for key in kwargs:
        if "data" in key:
            index = key.find("data")
            if key[index - 1] == "_" or key[index - 1] == "-":
                index -= 1
            label_list.append(key[:index])
            time_series_dict[key[:index]] = kwargs[key]
            label_counter += 1
    
    if label_counter == 0:
        raise Exception("No _data arguments passed to function")
    max_harm = 0
    all_harmonics = []
    colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
    c_counter=0
    for label in label_list:
        if "colour" not in time_series_dict[label]:
            if c_counter>=len(colours):
             c_counter=0

            time_series_dict[label]["colour"]=colours[c_counter]
            c_counter+=1
        if "lw" not in time_series_dict[label]:
            time_series_dict[label]["lw"]=1
        if "alpha" not in time_series_dict[label]:
            time_series_dict[label]["alpha"]=1
        if "linestyle" not in time_series_dict[label]:
            time_series_dict[label]["linestyle"]="-"
        if kwargs["xaxis"] == "DC_potential":
            time_series_dict[label]["xaxis"] =sci.get_DC_component(time_series_dict[label]["time"],time_series_dict[label]["potential"], time_series_dict[label]["current"])
        else:
            time_series_dict[label]["xaxis"] = time_series_dict[label][kwargs["xaxis"]]
        if "harmonics" not in time_series_dict[label]:
            time_series_dict[label]["harmonics"] = sci.maximum_availiable_harmonics(
                time_series_dict[label]["time"], time_series_dict[label]["current"]
            )
            #plt.plot(time_series_dict[label]["time"],time_series_dict[label]["potential"])
           
            calculated_harmonics=True
            #plt.show()
        else:
            calculated_harmonics=False
        max_harm = max([len(time_series_dict[label]["harmonics"]), max_harm])
        if kwargs["save_csv"] is not False:
            filename=label+".csv"
        else:
            filename=False
        harm_dict[label] = sci.plot.generate_harmonics(
            time_series_dict[label]["time"],
            time_series_dict[label]["current"],
            hanning=kwargs["hanning"],
            func=kwargs["fft_func"],
            one_sided=kwargs["one_sided"],
            harmonics=time_series_dict[label]["harmonics"],
            filter_val=kwargs["filter_val"],
            save_csv=filename
        )
        all_harmonics += time_series_dict[label]["harmonics"]
    all_harmonics = list(set(sorted(all_harmonics)))
    num_harmonics = len(all_harmonics)
    altering_harmonics=False
    if "axes_list" not in kwargs:
        fig, kwargs["axes_list"] = plt.subplots(num_harmonics, 1)
    elif calculated_harmonics==False:
        if len(kwargs["axes_list"]) != num_harmonics:
            raise ValueError(
                "Wrong number of axes for harmonics (calculated number of harmonics is {0}, provided axes is of length {1})".format(
                    num_harmonics, len(kwargs["axes_list"])
                )
            )
    elif len(kwargs["axes_list"]) != num_harmonics:
        print(
            "Warning: Wrong number of axes for harmonics (calculated number of harmonics is {0}, provided axes is of length {1})".format(
                num_harmonics, len(kwargs["axes_list"])
            )
        )
        
        print("Reducing length of harmonic range to match number of provided axes")
        all_harmonics=all_harmonics[:len(kwargs["axes_list"])]
        for plot_name in label_list:
            time_series_dict[plot_name]["harmonics"]=time_series_dict[plot_name]["harmonics"][:len(kwargs["axes_list"])]
        num_harmonics = len(all_harmonics)
    counter=0
    for plot_name in label_list:
        counter+=1
        freq=sci.get_frequency(time_series_dict[plot_name]["time"], time_series_dict[plot_name]["current"])
        if kwargs["h_num"] != False:
            if counter==1:
                for z in range(0, len(all_harmonics)):
                    ax = kwargs["axes_list"][z]
                    ax2 = ax.twinx()
                    ax2.set_yticks([])
                    if kwargs["h_num"]==True:
                        ax2.set_ylabel(all_harmonics[z], rotation=0)
                    elif kwargs["h_num"]=="frequencies":
                        text = ax.text(0.98, 0.5, "{0} Hz".format(int(all_harmonics[z]*freq)), transform=ax.transAxes,
                                        fontsize=8, bbox=dict( facecolor='white', alpha=0.5),
                                        va='bottom', ha='right')
                        
        

        harmonics = time_series_dict[plot_name]["harmonics"]
        if kwargs["title"] is not None:
            kwargs["axes_list"][0].set_title(kwargs["title"])
       
        if kwargs["clip_oscillations"] is not None:
            
            start=kwargs["clip_oscillations"]/freq
            end=time_series_dict[plot_name]["time"][-1]-start
            idx=np.where((time_series_dict[plot_name]["time"]>start) & (time_series_dict[plot_name]["time"]<end))
        else:
            idx=np.where(time_series_dict[plot_name]["time"]>=0)
        for i in range(0, len(harmonics)):
            ax = kwargs["axes_list"][i]
            xaxis = time_series_dict[plot_name]["xaxis"][idx]
            if i == 0:
                if i == harmonics[i]:
                    if kwargs["plot_func"] == np.abs or kwargs["plot_func"] == abs:
                        pf = np.real
                    else:
                        pf = kwargs["plot_func"]
                else:
                    pf = kwargs["plot_func"]
               

                ax.plot(
                    xaxis,
                    pf(harm_dict[plot_name][i, :])[idx],
                    label=plot_name,
                    alpha=time_series_dict[plot_name]["alpha"],
                    color=time_series_dict[plot_name]["colour"],
                    lw=time_series_dict[plot_name]["lw"],
                    linestyle=time_series_dict[plot_name]["linestyle"]
                )
            else:
                
                ax.plot(
                    xaxis,
                    kwargs["plot_func"](harm_dict[plot_name][i, :])[idx],
                    alpha=time_series_dict[plot_name]["alpha"],
                    color=time_series_dict[plot_name]["colour"],
                    lw=time_series_dict[plot_name]["lw"],
                )
            if kwargs["nyticks"] is not None:
                if plot_name==label_list[-1]:
                    print("setting", kwargs["nyticks"])
                    ax.yaxis.set_major_locator(plt.MaxNLocator(kwargs["nyticks"]))
                
            if i == ((num_harmonics) // 2):
                ax.set_ylabel(kwargs["ylabel"], labelpad=kwargs["ylabelpad"])
            if i == num_harmonics - 1:
                ax.set_xlabel(kwargs["xlabel"])
            else:
                if kwargs["remove_xaxis"]==True:
                    ax.set_xticks([])
            if i == 0:
                if kwargs["legend"] is not None:

                    ax.legend(**kwargs["legend"])
    return kwargs["axes_list"]
