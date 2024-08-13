import numpy as np
import Surface_confined_inference as sci
import copy

def top_hat_filter(times, time_series, **kwargs):
    """
    Args:
        times (list): list of timepoints with constant dt
        time_series (list): list of current values at times
    Returns:
        (list): list of filtered Fourier transform values
    The function extracts and returns positive harmonics in the form of the Fourier transform. The harmonics selected are dependent on the value
    of the `_internal_options.Fourier_harmonics` value, which is a list of increasing but not necessarily consecutive values. There are a number
    of ways in which the Fourier values can be returned, which is controlled by the `_internal_options.Fourier_function` value
    """
    if "Fourier_window" not in kwargs:
        kwargs["Fourier_window"]=False
    L = len(time_series)
    window = np.hanning(L)
    if kwargs["Fourier_window"] == "hanning":
        print("hanning")
        time_series = np.multiply(time_series, window)
    if "top_hat_width" not in kwargs:
        filter_val = 0.5
    else:
        filter_val=kwargs["top_hat_width"]
    if "Fourier_harmonics" in kwargs:
        harmonic_range=kwargs["Fourier_harmonics"]
    else:
        raise Exception("Need to define the harmonics you want to extract!")
    if "Fourier_function" not in kwargs:
        kwargs["Fourier_function"]="abs"
    frequencies = np.fft.fftfreq(len(time_series), times[1] - times[0])
    Y = np.fft.fft(time_series)

    true_harm=abs(frequencies[np.where(Y==max(Y))])[0]
    top_hat = copy.deepcopy(Y)
    
    
    if sum(np.diff(harmonic_range)) != len(harmonic_range) - 1:
        results = np.zeros(len(top_hat), dtype=complex)
        for i in range(0, len(harmonic_range)):

            true_harm_n = true_harm * harmonic_range[i]
            index = tuple(
                np.where(
                    (frequencies < (true_harm_n + (true_harm * filter_val)))
                    & (frequencies > true_harm_n - (true_harm * filter_val))
                )
            )
            filter_bit = top_hat[index]
            results[index] = filter_bit
    else:
        first_harm = (harmonic_range[0] * true_harm) - (true_harm * filter_val)
        last_harm = (harmonic_range[-1] * true_harm) + (true_harm * filter_val)
        freq_idx_1 = tuple(
            np.where((frequencies > first_harm) & (frequencies < last_harm))
        )
        likelihood_1 = top_hat[freq_idx_1]
        results = np.zeros(len(top_hat), dtype=complex)
        results[freq_idx_1] = likelihood_1
    if kwargs["Fourier_function"]== "abs":
        return abs(results)
    elif kwargs["Fourier_function"]== "imag":
        return np.imag(results)
    elif kwargs["Fourier_function"]== "real":
        return np.real(results)
    elif kwargs["Fourier_function"]== "composite":
        comp_results = np.append(np.real(results), np.imag(results))
        return comp_results
    elif kwargs["Fourier_function"]== "inverse":
        return np.fft.ifft(results)
