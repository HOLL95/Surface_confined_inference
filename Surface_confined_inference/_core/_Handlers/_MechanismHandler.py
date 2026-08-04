"""Handler that simulates a YAML reaction mechanism instead of the C++ models.

Selected by `model="mechanism"`, with the network named by the `mechanism`
option. The ODE system, the DiffSL source and the compiled solver are all
generated from that network by `_MechanismProcess`; this module is only the
adapter onto the `BaseHandler` contract.

Everything here works on Surface_confined_inference's nondimensional scales.
That is not a convenience: the generated rate law is `exp(sum_k c_k * op^k)`
with `op = E - E0 - I*Ru`, and the Marcus integral it is fitted to assumes the
overpotential is in units of RT/F. Feeding it dimensional volts would be wrong
by a factor of F/RT (~39 at 298 K), so `ParameterInterface.nondimensionalise`
is doing load-bearing work rather than bookkeeping.
"""
import copy
import warnings
import pydiffsol as ds
import numpy as np
import sympy

from ._BaseVoltammetry import ContinuousHandler
from .._Generic._MechanismHelpers import waveform
from .._Generic._MechanismProcess import (
    build_odes,
    build_solver,
    load_network,
    mechanism_source,
    render_diffsl,
    render_potential,
    required_parameters,
)

#Surface_confined_inference's input names -> the names the generated waveform
#uses. NDParams.omega_nondim already folds in the 2*pi, which is what the
#generated sin(omega*t + phase) wants, so this is a pure rename.
_WAVEFORM_NAMES = {
    "FTACV": {
        "E_start": "Estart",
        "E_reverse": "Ereverse",
        "v": "v",
        "delta_E": "Eamp",
        "omega": "omega",
        "phase": "phase",
    },
    "DCV": {"E_start": "Estart", "E_reverse": "Ereverse", "v": "v"},
    #The square wave inputs are already named after the waveform, so `smoothing`
    #is the only rename left: `alpha` is the gate width in the generated model,
    #but it is the electron transfer symmetry factor everywhere else.
    "SWVtanh":{"omega":"omega",
                "Estart":"Estart",
                "Estep":"Estep",
                "Estop":"Estop",
                "Eamp":"Eamp",
                "smoothing":"alpha"}
}



#Compiled models are shared across handler instances: SingleExperiment rebuilds
#its handler on every optim_list/fixed_parameters/option change, and generating
#plus compiling the DiffSL each time would dominate an optimisation loop.
_COMPILED_CACHE = {}


def _cache_key(
    mechanism, experiment_type, poly_degree, solver, nd_inputs, times, max_resistance
):
    #max_resistance sets the IR-drop margin on the fitted overpotential window,
    #so it belongs in the key. Without it a scan over Ru reuses whichever window
    #compiled first and every later value is fitted over the wrong range. The E0
    #shift needs no entry: the window is bounded by the potential range, which
    #already lives here via nd_inputs.
    return (
        repr(sorted(mechanism.items(), key=lambda kv: kv[0])),
        experiment_type,
        poly_degree,
        solver,
        tuple(sorted((k, float(v)) for k, v in nd_inputs.items())),
        len(times),
        float(times[0]),
        float(times[-1]),
        max_resistance,
    )


class MechanismHandler(ContinuousHandler):
    """Simulate a generated mechanism over a continuous voltammetric waveform.

    Inherits `calculate_times` and the waveform bookkeeping from
    ContinuousHandler; overrides everything that would otherwise reach for the
    C++ solver.

    Methods:
        - simulate(sim_params, times)
        - get_thetas(sim_params, times)
        - get_voltage(times, input_parameters, validation_parameters)
        - dispersion_simulator(sim_params, times)
        - _get_parallel_function()
    """

    def __init__(self, options, param_interface):
        super().__init__(options, param_interface)
        if options.mechanism is None:
            raise ValueError("model='mechanism' needs the `mechanism` option set")
        self._mechanism = mechanism_source(options.mechanism)
        self._network = load_network(self._mechanism)
        #Only these reach the solver; sim_dict carries entries the generated
        #model has no slot for (CdlE1-3, phase, area, Temp ...).
        self._known = set(required_parameters(self._network))

    """
    Model construction
    """

    def _waveform_inputs(self, nd_dict):
        """Select and rename the nondimensional waveform parameters."""
        names = _WAVEFORM_NAMES[self.options.experiment_type]
        missing = [x for x in names if x not in nd_dict]
        if missing:
            raise KeyError(
                "{0} needs {1}, missing {2}".format(
                    self.options.experiment_type, sorted(names), missing
                )
            )
        return {names[key]: float(nd_dict[key]) for key in names}

    def _initial_state(self, nd_dict, nd_inputs, state_names):
        """Initial coverages and current.

        `reference` matches the C++ model, which starts fully reduced with the
        capacitive current already flowing (ODE_simulator_sundials.cpp:101).
        `equal_split` is the generated model's own convention: coverage shared
        evenly across the species, current starting at zero. The two give
        visibly different first cycles, so they are not interchangeable.

        Only the integrated states get an initial value. The species eliminated
        by the conservation sum is not one of them -- its coverage is whatever
        the others leave over.
        """
        states = [x for x in state_names if x != "I"]
       
        #Shared over every species, including the eliminated one, so the
        #implied coverage of that species is the remainder.
        share = 1 / len(self._network.species)
        return dict([(x, share) for x in states] + [("I", 0.0)])
        #dE/dt at t=0+, analytically. Differentiating the waveform instead picks
        #up a DiracDelta from the Heaviside gate that opens at t=0.
        dE0 = nd_inputs["v"]
        if self.options.experiment_type == "FTACV":
            dE0 += nd_inputs["Eamp"] * nd_inputs["omega"] * np.cos(nd_inputs["phase"])
        return dict(
            [(x, 0.0) for x in states] + [("I", nd_dict["Cdl"] * dE0)]
        )

    def _compiled(self, nd_dict, times):
        """Get or build the compiled model for this waveform and time grid."""
        nd_inputs = self._waveform_inputs(nd_dict)
        max_resistance = nd_dict["Ru"] if nd_dict["Ru"] > 0 else 1.0
        key = _cache_key(
            self._mechanism,
            self.options.experiment_type,
            self.options.poly_degree,
            self.options.diffsl_solver,
            nd_inputs,
            times,
            float(max_resistance),
        )
        if key not in _COMPILED_CACHE:
            
            expression = waveform(self.options.experiment_type, params=nd_inputs)
            odes = build_odes(self._network)
            potential = render_potential(expression.subs(nd_inputs))
            model = render_diffsl(
                self._network, odes, potential, poly_degree=self.options.poly_degree
            )
            _COMPILED_CACHE[key] = build_solver(
                model,
                times,
                expression,
                nd_inputs,
                #The IR-drop bound widens the fitted overpotential window, so it
                #has to be on the same nondimensional scale as everything else.
                max_current=1.0,
                max_resistance=max_resistance,
                ode_solver=getattr(ds.OdeSolverType, self.options.diffsl_solver),
            )
        return _COMPILED_CACHE[key], nd_inputs

    """
    Solving
    """

    def _states(self, nd_dict, times):
        """Solve, returning every state row on the requested time grid."""
        compiled, nd_inputs = self._compiled(nd_dict, times)
        layout = compiled.layout.with_initials(
            self._initial_state(nd_dict, nd_inputs, compiled.state_names)
        )

        values = {key: nd_dict[key] for key in nd_dict if key in self._known}
        
        vector = layout.pack(values)
        # The ramp waveforms are built from Heaviside gates, so FTACV and DCV both
        # take the stepped stop_i/reset_i path. calculate_times stops a little
        # short of the final switchpoint, which lands in a window where the event
        # handling diverges (62.0 and 62.11 fail where 62.2 succeeds on the same
        # model). Integrating to the switchpoint is well behaved, and the extra
        # point is appended at the end, so the retained values are exact solver
        # output rather than interpolated. SWVtanh has no switchpoint (and no
        # Ereverse): its gates are smooth, so it goes straight through.
        grid = np.asarray(times)
        if self.options.experiment_type in ("FTACV", "DCV"):
            t_switch = (
                2
                * abs(nd_inputs["Ereverse"] - nd_inputs["Estart"])
                / nd_inputs["v"]
            )
            if 0.9 * t_switch < grid[-1] < t_switch:
                grid = np.append(grid, t_switch)
        solution = compiled.ode.solve_dense(vector, grid)
        return np.asarray(solution.ys)[:, : len(times)]

    def simulate(self, sim_params, times):
        """Simulate the mechanism.

        Args:
            sim_params (dict): parameter values, dimensional
            times (array-like): nondimensional time points

        Returns:
            numpy.ndarray: total current at `times`
        """
        if self.options.dispersion == True:
            return self.dispersion_simulator(sim_params, times)
        nd_dict = self.param_interface.nondimensionalise(sim_params)
        return self._states(nd_dict, times)[self._current_row()]

    def _current_row(self):
        """Row of the output block holding the requested current.

        The generated model reports the species, then the total current, then
        the Faradaic current gamma*dtheta. Only the first two are states -- the
        Faradaic current is an out_i entry, evaluated at each output point.
        """
        return -1 if self.options.Faradaic_only == True else -2

    def get_thetas(self, sim_params, times):
        """Return every output row: the independent species, the total current,
        then the Faradaic current."""
        nd_dict = self.param_interface.nondimensionalise(sim_params)
        return self._states(nd_dict, times)

    def get_voltage(self, times, input_parameters, validation_parameters):
        """Potential at `times`, evaluated from the waveform actually integrated.

        Args:
            times (array-like): nondimensional time points
            input_parameters (dict): dimensional input parameters
            validation_parameters (dict): checked against, as in the base class

        Returns:
            numpy.ndarray: nondimensional potential
        """
        inputs = super(ContinuousHandler, self).get_voltage(
            times, input_parameters, validation_parameters
        )
        nd_inputs = self._waveform_inputs(
            self.param_interface.nondimensionalise(
                {key: inputs[key] for key in inputs if key in self.param_interface.sim_dict}
            )
        )
        import time
        start=time.time()
        expression = waveform(self.options.experiment_type, params=nd_inputs)
        
        E_func = sympy.lambdify(
            sympy.symbols("t"), expression.subs(nd_inputs), modules="numpy"
        )
        return np.asarray(E_func(np.asarray(times)), dtype=float)

    """
    Dispersion
    """

    def dispersion_simulator(self, sim_params, times):
        """Sequential quadrature over the dispersed parameters.

        The base implementation fans out over multiprocessing, which needs the
        per-simulation callable to pickle. A compiled pydiffsol Ode does not, so
        this runs in-process regardless of `parallel_cpu`. The compiled model is
        shared across the quadrature points, so the cost is the solves alone.
        """
        if self.options.parallel_cpu > 1:
            warnings.warn(
                "model='mechanism' evaluates dispersion sequentially; the "
                "compiled solver cannot be sent to worker processes"
            )
        mutable_dict = copy.deepcopy(self.param_interface.sim_dict)
        for key in sim_params.keys():
            mutable_dict[key] = sim_params[key]
        disp_params, values, weights = self.param_interface.disp_function(
            mutable_dict, self.param_interface.gh_values
        )

        total = None
        for i in range(len(weights)):
            for j, param in enumerate(disp_params):
                sim_params[param] = values[i][j]
            nd_dict = self.param_interface.nondimensionalise(sim_params)
            current = self._states(nd_dict, times)[self._current_row()] * np.prod(
                weights[i]
            )
            total = current if total is None else total + current
        return total

    def _get_parallel_function(self):
        raise NotImplementedError(
            "model='mechanism' does not use the multiprocessing dispersion path"
        )
