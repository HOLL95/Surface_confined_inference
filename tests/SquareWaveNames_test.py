"""The square wave input names, and the converter for the ones they replaced."""
import numpy as np
import pytest

import Surface_confined_inference as sci
from Surface_confined_inference import SingleExperiment

CURRENT_NAMES = {
    "omega": 10,
    "Estart": -0.4,
    "Estop": 0.3,
    "Estep": 2e-3,
    "Eamp": 10e-3,
    "sampling_factor": 200,
    "Temp": 278,
    "area": 0.07,
    "Surface_coverage": 1e-10,
}
LEGACY_NAMES = {
    "omega": 10,
    "E_start": -0.4,
    "delta_E": 0.7,
    "scan_increment": 2e-3,
    "SW_amplitude": 10e-3,
    "sampling_factor": 200,
    "v": 1,
    "N_elec": 1,
    "Temp": 278,
    "area": 0.07,
    "Surface_coverage": 1e-10,
}


def build(params):
    experiment = SingleExperiment("SquareWave", params)
    experiment.fixed_parameters = {"Cdl": 1e-4, "gamma": 1e-10, "alpha": 0.5, "Ru": 100}
    experiment.boundaries = {"E0": [0, 0.2], "k0": [1e-3, 1000]}
    experiment.optim_list = ["E0", "k0"]
    return experiment


class TestLegacyConversion:
    def test_current_names_are_not_converted(self):
        assert (
            sci.convert_legacy_square_wave_params(CURRENT_NAMES, "SquareWave")
            is CURRENT_NAMES
        )

    def test_legacy_names_warn(self):
        with pytest.warns(DeprecationWarning, match="renamed onto the SWVtanh set"):
            converted = sci.convert_legacy_square_wave_params(LEGACY_NAMES, "SquareWave")
        assert converted == CURRENT_NAMES

    def test_other_experiments_keep_delta_E_and_v(self):
        #delta_E is the AC amplitude and v the scan rate for FTACV, so the
        #square wave conversion must not touch them.
        ftacv = {"E_start": -0.4, "delta_E": 0.15, "v": 25e-3}
        assert sci.convert_legacy_square_wave_params(ftacv, "FTACV") is ftacv

    def test_cathodic_direction_comes_from_the_sign_of_v(self):
        legacy = dict(LEGACY_NAMES, E_start=0.3, v=-1)
        with pytest.warns(DeprecationWarning):
            converted = sci.convert_legacy_square_wave_params(legacy, "SquareWave")
        assert converted["Estart"] == 0.3
        assert converted["Estop"] == -0.4

    @pytest.mark.parametrize(
        "params, message",
        [
            (dict(LEGACY_NAMES, Estart=-0.4), "both Estart and the name it replaced"),
            (dict(LEGACY_NAMES, Estop=0.3), "both Estop and the potential window"),
            ({"E_start": -0.4, "delta_E": 0.7, "v": 0}, "v=0 cannot be converted"),
            ({"delta_E": 0.7}, "needs Estart as well"),
            ({"Estart": -0.4, "Estop": 0.3, "v": -1}, "disagrees with the direction"),
            ({"E_start": -0.4, "N_elec": 2}, "only implements a single"),
        ],
    )
    def test_unconvertible_inputs_raise(self, params, message):
        with pytest.raises(ValueError, match=message):
            sci.convert_legacy_square_wave_params(params, "SquareWave")

    def test_legacy_invocation_simulates_identically(self):
        current = build(CURRENT_NAMES)
        with pytest.warns(DeprecationWarning):
            legacy = build(LEGACY_NAMES)
        assert legacy._internal_options.input_params == current._internal_options.input_params
        times = current.calculate_times()
        assert np.array_equal(times, legacy.calculate_times())
        assert np.array_equal(
            np.array(current.simulate([0.1, 100], times)),
            np.array(legacy.simulate([0.1, 100], times)),
        )
        assert np.array_equal(
            current.get_voltage(times), legacy.get_voltage(times)
        )


class TestScanDirection:
    def test_direction_follows_estop(self):
        anodic = build(CURRENT_NAMES)
        cathodic = build(dict(CURRENT_NAMES, Estart=0.3, Estop=-0.4))
        anodic_potentials = anodic._ExperimentHandler.SW_params["E_p"]
        cathodic_potentials = cathodic._ExperimentHandler.SW_params["E_p"]
        assert anodic_potentials[0] == pytest.approx(-0.4)
        assert cathodic_potentials[0] == pytest.approx(0.3)
        assert np.all(np.diff(anodic_potentials) > 0)
        assert np.all(np.diff(cathodic_potentials) < 0)
        assert len(anodic_potentials) == len(cathodic_potentials)

    def test_empty_potential_window_raises(self):
        with pytest.raises(ValueError, match="no staircase to walk"):
            build(dict(CURRENT_NAMES, Estop=CURRENT_NAMES["Estart"]))
