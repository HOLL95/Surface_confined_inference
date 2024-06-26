import numpy as np
import copy
import matplotlib.pyplot as plt
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
                harmonics[i, :] = 2 * ((np.fft.ifft(top_hat)))

            else:
                harmonics[i, :] = np.fft.ifft(f_domain_harmonic)
        else:

            ft_peak_return[i, :] = f_domain_harmonic

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
        kwargs["legend"] = {"loc": "center"}
    if "h_num" not in kwargs:
        kwargs["h_num"] = True
    if "colour" not in kwargs:
        kwargs["colour"] = None
    if "lw" not in kwargs:
        kwargs["lw"] = 1
    if "alpha" not in kwargs:
        kwargs["alpha"] = 1
    if "one_sided" not in kwargs:
        kwargs["one_sided"] = True
    if "filter_val" not in kwargs:
        kwargs["filter_val"] = 0.5

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
        Exception("No _data arguments passed to function")
    max_harm = 0
    all_harmonics = []
    for label in label_list:

        if kwargs["xaxis"] == "DC_potential":
            time_series_dict[label]["potential"]
            fft_pot = np.fft.fft(pot)
            fft_freq = np.fft.fftfreq(
                len(pot),
                time_series_dict[label]["time"][1] - time_series_dict[label]["time"][0],
            )
            max_freq = sci.get_frequency(
                time_series_dict[label]["time"], time_series_dict[label]["current"]
            )
            zero_harm_idx = np.where(
                (fft_freq > -(0.5 * max_freq)) & (fft_freq < (0.5 * max_freq))
            )
            dc_pot = np.zeros(len(fft_pot), dtype="complex")
            dc_pot[zero_harm_idx] = fft_pot[zero_harm_idx]
            time_series_dict[label]["xaxis"] = np.real(np.fft.ifft(dc_pot))
        else:
            time_series_dict[label]["xaxis"] = time_series_dict[label][kwargs["xaxis"]]
        if "harmonics" not in kwargs:
            time_series_dict[label]["harmonics"] = sci.maximum_availiable_harmonics(
                time_series_dict[label]["time"], time_series_dict[label]["current"]
            )
        elif "harmonics" not in time_series_dict[label]:
            time_series_dict[label]["harmonics"] = kwargs["harmonics"]
        max_harm = max([len(time_series_dict[label]["harmonics"]), max_harm])
        harm_dict[label] = sci.plot.generate_harmonics(
            time_series_dict[label]["time"],
            time_series_dict[label]["current"],
            hanning=kwargs["hanning"],
            func=kwargs["fft_func"],
            one_sided=kwargs["one_sided"],
            harmonics=time_series_dict[label]["harmonics"],
            filter_val=kwargs["filter_val"],
        )
        all_harmonics += time_series_dict[label]["harmonics"]
    all_harmonics = list(set(sorted(all_harmonics)))
    num_harmonics = max_harm
    if "axes_list" not in kwargs:
        fig, kwargs["axes_list"] = plt.subplots(num_harmonics, 1)
    elif len(kwargs["axes_list"]) != num_harmonics:
        raise ValueError(
            "Wrong number of axes for harmonics (calculated number of harmonics is {0}, provided axes is of length {1})".format(
                num_harmonics, len(kwargs["axes_list"])
            )
        )
    if kwargs["h_num"] != False:
        for i in range(0, len(all_harmonics)):
            ax = kwargs["axes_list"][i]
            ax2 = ax.twinx()
            ax2.set_yticks([])
            ax2.set_ylabel(all_harmonics[i], rotation=0)
    for plot_name in label_list:
        harmonics = time_series_dict[plot_name]["harmonics"]
        for i in range(0, len(harmonics)):
            ax = kwargs["axes_list"][i]
            xaxis = time_series_dict[plot_name]["xaxis"]
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
                    pf(harm_dict[plot_name][i, :]),
                    label=plot_name,
                    alpha=kwargs["alpha"],
                    color=kwargs["colour"],
                    lw=kwargs["lw"],
                )
            else:
                ax.plot(
                    xaxis,
                    kwargs["plot_func"](harm_dict[plot_name][i, :]),
                    alpha=kwargs["alpha"],
                    color=kwargs["colour"],
                    lw=kwargs["lw"],
                )
            if i == ((num_harmonics) // 2):
                ax.set_ylabel(kwargs["ylabel"])
            if i == num_harmonics - 1:
                ax.set_xlabel(kwargs["xlabel"])
            if i == 0:
                if kwargs["legend"] is not None:

                    ax.legend(**kwargs["legend"])
