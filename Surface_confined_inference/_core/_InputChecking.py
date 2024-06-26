import numpy as np
import Surface_confined_inference as sci


def check_input_dict(input_dict, parameters):
    user_params = set(input_dict.keys())
    required_params = set(parameters)
    extra = user_params - required_params
    missing = required_params - user_params
    if len(missing) > 0:
        raise ValueError(
            "Simulation requires the following parameters: {0}".format(
                (" ").join(list(missing))
            )
        )
    if len(extra) > 0:
        raise ValueError(
            "The following parameters are not required for the simulation: {0}".format(
                (" ").join(list(extra))
            )
        )


def get_frequency(times, current):
    fft = abs(np.fft.fft(current))
    frequency = np.fft.fftfreq(len(current), times[1] - times[0])
    max_freq = abs(max(frequency[np.where(fft == max(fft))]))
    return max_freq


def maximum_availiable_harmonics(times, current):
    fft = abs(np.fft.fft(current))
    frequencies = np.fft.fftfreq(len(current), times[1] - times[0])
    input_freq = sci.get_frequency(times, current)
    in_noise = False
    max_found = 1
    while in_noise == False:
        index = max_found * input_freq
        noise_level = np.mean(
            fft[
                np.where(
                    (frequencies > (index - 0.5 * input_freq))
                    & (frequencies < (index - 0.4 * input_freq))
                )
            ]
        )
        peak_level = max(
            fft[
                np.where(
                    (frequencies > (index - 0.1 * input_freq))
                    & (frequencies < (index + 0.1 * input_freq))
                )
            ]
        )
        if peak_level > (3 * noise_level):
            max_found += 1
        else:
            in_noise = True
    return list(range(0, max_found))
