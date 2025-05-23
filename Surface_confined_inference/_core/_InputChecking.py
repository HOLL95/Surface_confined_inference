import numpy as np
import Surface_confined_inference as sci


def check_input_dict(input_dict, parameters, **kwargs):
    if "optional_arguments" not in kwargs:
        kwargs["optional_arguments"]=set()
    else:
        kwargs["optional_arguments"]=set(kwargs["optional_arguments"])
    user_params = set(input_dict.keys())
    required_params = set(parameters)
    missing = required_params - user_params
    
    extra = user_params - required_params
    if len(missing) > 0:
        raise ValueError(
            "Simulation requires the following parameters: {0}".format(
                (" ").join(list(missing))
            )
        )
    if len(extra) > 0:
       
        if len(extra.intersection(kwargs["optional_arguments"])) ==len(extra):
            
            pass
        else:
            raise ValueError(
                "The following parameters are not required for the simulation: {0}".format(
                    (" ").join([x for x in list(extra) if x not in kwargs["optional_arguments"]])
                )
        )


def get_frequency(times, current):
    fft = abs(np.fft.fft(current))
    frequency = np.fft.fftfreq(len(current), times[1] - times[0])
    max_freq = abs(max(frequency[np.where(fft == max(fft))]))
    return max_freq

def get_DC_component(time, potential, current):
    pot=potential
    fft_pot = np.fft.fft(pot)
    fft_freq = np.fft.fftfreq(
        len(pot),
        time[1] - time[0],
    )
    max_freq = sci.get_frequency(
        time, current
    )
    zero_harm_idx = np.where(
        (fft_freq > -(0.5 * max_freq)) & (fft_freq < (0.5 * max_freq))
    )
    dc_pot = np.zeros(len(fft_pot), dtype="complex")
    dc_pot[zero_harm_idx] = fft_pot[zero_harm_idx]
    return np.real(np.fft.ifft(dc_pot))


def maximum_availiable_harmonics(times, current):
    import matplotlib.pyplot as plt
    fft = abs(np.fft.fft(current))
    frequencies = np.fft.fftfreq(len(current), times[1] - times[0])
    input_freq = sci.get_frequency(times, current)
    in_noise = False
    max_found = 1
    #plt.plot(times, current)
    #plt.show()
    #plt.plot(frequencies, fft)
    #plt.show()
    #fig, ax=plt.subplots()
    while in_noise == False:
        index = max_found * input_freq
        noise_idx= np.where(
                    (frequencies > (index + 0.2 * input_freq))
                    & (frequencies < (index + 0.5 * input_freq))
                )
        peak_idx=np.where(
                    (frequencies > (index - 0.2 * input_freq))
                    & (frequencies < (index + 0.2 * input_freq))
                )
        
        noise_level = np.mean(np.abs(fft[noise_idx]))
        peak_level = max(np.abs(fft[peak_idx]))
        
        
        #print(noise_level, peak_level, peak_level/noise_level)
        #print(frequencies[noise_idx], (index - 0.2 * input_freq), (index + 0.2 * input_freq))
        #plt.plot(frequencies, fft)
        #plt.plot(frequencies[peak_idx], fft[peak_idx])
        #plt.plot(frequencies[noise_idx], fft[noise_idx])
        #plt.show()
        if peak_level > (1.5 * noise_level):
            max_found += 2
        else:
            in_noise = True
    #ax.legend()
    #plt.show()
    return list(range(0, max_found+2))
