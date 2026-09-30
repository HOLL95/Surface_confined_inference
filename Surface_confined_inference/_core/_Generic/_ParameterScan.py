
import os

import numpy as np

from .._Options._SingleExperimentOptions import SingleExperimentOptions
from ._MechanismProcess import load_network, mechanism_source, required_parameters

#Sweep bounds per parameter family, matched against the parameter name by
#substring: `k0` covers k0_1, k0_2 ..., `kcat` covers kcatf_1/kcatb_1. All
#dimensional -- `lambda` is volts here, which is what the e_nondim branch of
#NDParams.construct_function_dict expects for a reorganisation energy.
_RANGE_VALUES = {
    "k0": [0.1, 1000],
    "Ru": [0.1, 1000],
    "E0": [-1, 1],
    "lambda": [0.1, 1.5],
    "kcat": [100, 1e5],
    "gamma": [1e-11, 1e-10],
    "Cdl": [1e-5, 1e-4],
}

#Spread over more than this many decades and the sweep is logarithmic. k0, Ru and
#kcat span four or three decades; Cdl and gamma span one, so they stay linear.
_LOG_DECADES = 1.5

_MODES = ("individual", "pairwise")

#The cell parameters every experiment declares. Listed last in the generated
#input dictionary so the waveform parameters read together at the top.
_CELL_INPUTS = ("area", "Temp", "Surface_coverage")

#Multiplicative perturbations applied to each input parameter in turn by the
#input scan, matching the show_input_scan diagnostic in the mechanism class.
_INPUT_SCALARS = (0.75, 1, 1.5)

_LIBRARIES = (
    ("numpy", "np"),
    ("matplotlib.pyplot", "plt"),
    ("copy", "copy"),
    ("itertools", "it"),
    ("multiprocessing", "mp"),
    ("os", "os"),
)

#Where the generated script writes its figures when the caller does not name a
#directory. Relative, so it lands beside wherever the script is run from.
_FIGURE_DIRECTORY = "parameter_scan_figures"

#Saved figures are named <what was swept>.png under that directory; the input
#scan draws only one figure, so its name is fixed.
_INPUT_SCAN_FIGURE = "input_scan"

#Figure sizes in inches, (width, height), for the total current figures and for
#the harmonic figures drawn alongside them.
_CURRENT_FIGSIZE = (8, 8)
_HARMONICS_FIGSIZE = (6, 9)

#Harmonics plotted when the caller does not choose, for experiments with a
#sinusoidal component.
_DEFAULT_HARMONICS = list(range(1, 9))


def _sweep(name, range_size):
    """Source for one parameter's sweep, and the value to hold it at otherwise.

    Returns:
        tuple: (expression string, midpoint value), or (None, None) if the name
            matches no known family
    """
    for family, (lo, hi) in _RANGE_VALUES.items():
        if family not in name:
            continue
        logarithmic = (
            lo > 0 and hi > 0 and np.log10(hi) - np.log10(lo) > _LOG_DECADES
        )
        if logarithmic:
            loglo, loghi = np.log10(lo), np.log10(hi)
            expression = "np.logspace({0}, {1}, range_size)".format(loglo, loghi)
            values = np.logspace(loglo, loghi, range_size)
        else:
            expression = "np.linspace({0}, {1}, range_size)".format(lo, hi)
            values = np.linspace(lo, hi, range_size)
        return expression, float(values[range_size // 2])
    return None, None


def _input_names(experiment_type, potential_input):
    """Input-potential parameter names the chosen experiment declares.

    Read off the options class rather than hardcoded per experiment, so a new
    experiment type is picked up without touching this module.

    Raises:
        ValueError: for an unknown experiment type, or for Generic without an
            expression to drive it
    """
    classes = SingleExperimentOptions._experiment_classes
    if experiment_type not in classes:
        raise ValueError(
            "experiment_type must be one of {0}, not {1!r}".format(
                sorted(classes), experiment_type
            )
        )
    option = classes[experiment_type].input_params
    if experiment_type == "Generic":
        if potential_input is None:
            raise ValueError(
                "experiment_type='Generic' needs `potential_input`, a sympy "
                "expression for the potential"
            )
        names = set(option.required_keys or ()) | set(_CELL_INPUTS)
    else:
        names = set(option.target)
    #Waveform parameters first, then the cell parameters, each alphabetically.
    waveform = sorted(names - set(_CELL_INPUTS))
    return waveform + [x for x in _CELL_INPUTS if x in names]


def _positive_numbers(values):
    """True for a sequence of numbers, all above zero (bool excluded)."""
    try:
        return all(
            isinstance(x, (int, float, np.integer, np.floating))
            and not isinstance(x, bool)
            and x > 0
            for x in values
        )
    except TypeError:
        return False


def _oscillates(input_names, potential_input):
    """Whether the waveform has a sinusoid, and so harmonics worth plotting.

    Taken from an `omega` input parameter: FTACV and PSV declare one, DCV and
    the square-wave experiments do not. A Generic experiment declares only the
    cell parameters, so its potential expression is checked for the symbol
    instead.
    """
    if "omega" in input_names:
        return True
    if potential_input is None:
        return False
    symbols = getattr(potential_input, "free_symbols", None)
    if symbols is None:
        import sympy

        symbols = sympy.sympify(potential_input).free_symbols
    return "omega" in {str(x) for x in symbols}


def _mechanism_literal(mechanism):
    """How the generated script should refer back to the mechanism.

    A path is emitted as-is so the script stays readable; anything else is
    inlined as the parsed mapping, since there is no file for it to name.
    """
    if isinstance(mechanism, (str, os.PathLike)) and os.path.isfile(mechanism):
        return repr(os.fspath(mechanism))
    return repr(mechanism_source(mechanism))


def _dict_block(name, entries, comment="", headings=None):
    """Render `name = {...}`, one `"key": value` per line.

    `headings` maps a key to a comment line written just above it. A mechanism
    names its parameters `k0_1`, `lambda_1`, `E0_1` after the reaction index
    alone, which says nothing about which step they belong to, so the fixed
    parameters are grouped under the reaction equation as the file writes it.
    """
    headings = headings or {}
    lines = []
    for index, (key, value) in enumerate(entries):
        if key in headings:
            #Blank line between groups, but not before the first one -- the
            #opening brace already separates it.
            if lines:
                lines.append("")
            lines.append("\t#{0}".format(headings[key]))
        separator = "," if index < len(entries) - 1 else ""
        lines.append('\t"{0}": {1}{2}'.format(key, value, separator))
    return "{0} = {{{1}\n{2}\n}}".format(name, comment, "\n".join(lines))


def _input_scan():
    """The input-parameter diagnostic: what each waveform parameter does.

    Every input parameter is perturbed on its own and the resulting potential is
    plotted, one subplot per parameter. The timebase is rebuilt for each
    perturbed value, because duration and sample spacing both follow the
    waveform -- `omega`, `v` and `Estep` all change how long the experiment runs
    for (ContinuousHandler.calculate_times).

    calculate_times and get_voltage each take the perturbed dictionary as an
    argument, so the experiment itself is never mutated and the sweep that
    follows is unaffected. Both nondimensionalise against the constants the
    experiment was built with, so the two agree and dim_t/dim_e give back true
    seconds and volts.
    """
    return [
        "if potential_scan:",
        "\tcols = int(np.ceil(np.sqrt(len(scanned_inputs))))",
        "\trows = int(np.ceil(len(scanned_inputs) / cols))",
        "\tfig, axes = plt.subplots(rows, cols, squeeze=False)",
        "\taxes = axes.flatten()",
        "\tfor i, name in enumerate(scanned_inputs):",
        "\t\tfor scalar in input_scan_scalars:",
        "\t\t\tperturbed = {**input_parameter_values,",
        "\t\t\t\tname: input_parameter_values[name] * scalar}",
        "\t\t\tscan_times = experiment.calculate_times(",
        "\t\t\t\tinput_parameters=perturbed, dimensional=False)",
        "\t\t\tpotential = experiment.dim_e(",
        "\t\t\t\texperiment.get_voltage(scan_times, input_parameters=perturbed))",
        "\t\t\taxes[i].plot(experiment.dim_t(scan_times), potential,",
        "\t\t\t\tlabel=f'{name}={perturbed[name]:.3g}')",
        "\t\taxes[i].set_xlabel('Time (s)')",
        "\t\taxes[i].set_ylabel('Potential (V)')",
        "\t\taxes[i].legend()",
        "\tfor spare in axes[len(scanned_inputs):]:",
        "\t\tspare.set_axis_off()",
        "\tfig.tight_layout()",
        "\tfinish_figure(fig, '{0}')".format(_INPUT_SCAN_FIGURE),
        "\tshow_figures()",
    ]


def _loop(mode):
    """The sweep itself.

    Every simulation goes through `run_sweep`, so the points behind one figure
    are gathered first and drawn once they are all back. That keeps the pool to
    one batch per figure -- `range_size` points in individual mode, the whole
    `range_size` by `range_size` grid in pairwise -- rather than a call each.

    The swept value is written into fixed_parameters and optim_list is left
    empty, rather than the other way round. Assigning either one rebuilds the
    parameter handler and revalidates against the other as it currently stands,
    so a swept parameter moved from fixed_parameters to optim_list is briefly in
    neither ("either need to be set in optim_list, or set at a value using the
    fixed_parameters variable") or briefly in both (a warning, and the fixed
    value silently ignored). Keeping every parameter in fixed_parameters
    throughout has no such window. The rebuild per value is cheap: the compiled
    model is cached across handler instances.
    """
    if mode == "individual":
        return [
            "for parameter in parameter_ranges.keys():",
            "\tfig, ax = plt.subplots(figsize=current_figsize)",
            "\tvalues = parameter_ranges[parameter]",
            "\tcurrents = run_sweep([{parameter: value} for value in values])",
            "\tfor value, current in zip(values, currents):",
            "\t\tax.plot(x_vals, current, label=f'{parameter}={value}')",
            "\tax.set_xlabel(x_label)",
            "\tax.set_ylabel('Current (A)')",
            "\tax.legend()",
            "\tfinish_figure(fig, parameter)",
            "\tif harmonics is not None:",
            "\t\tfig, axes = plt.subplots(len(harmonics), 1,",
            "\t\t\tfigsize=harmonics_figsize, squeeze=False)",
            "\t\tplot_harmonic_set(axes[:, 0], parameter, zip(values, currents),",
            "\t\t\tylabel='Current (A)')",
            "\t\tfinish_figure(fig, f'{parameter}_harmonics')",
            "show_figures()",
        ]
    return [
        "combinations = list(it.combinations(parameter_ranges.keys(), 2))",
        "grid = list(it.product(range(range_size), range(range_size)))",
        "for param1, param2 in combinations:",
        "\tfig, ax = plt.subplots(range_size, range_size, figsize=current_figsize)",
        "\tcurrents = run_sweep([{param1: parameter_ranges[param1][j],",
        "\t\tparam2: parameter_ranges[param2][q]} for j, q in grid])",
        "\tfor (j, q), current in zip(grid, currents):",
        "\t\tvalue2 = parameter_ranges[param2][q]",
        "\t\tax[j, q].plot(x_vals, current, label=f'{param2}={value2}')",
        "\tfor j in range(0, range_size):",
        "\t\tax[j, 0].set_ylabel(f'{param1}={parameter_ranges[param1][j]}')",
        "\tfor q in range(0, range_size):",
        "\t\tax[-1, q].set_xlabel(x_label)",
        "\tfig.suptitle(f'{param1} vs {param2}')",
        "\tplt.tight_layout()",
        "\tfinish_figure(fig, f'{param1}_vs_{param2}')",
        "\tif harmonics is not None:",
        "\t\t#One column per value of param1, each overlaying every param2.",
        "\t\tby_point = dict(zip(grid, currents))",
        "\t\tfig, axes = plt.subplots(len(harmonics), range_size,",
        "\t\t\tfigsize=harmonics_figsize, squeeze=False)",
        "\t\tfor j in range(0, range_size):",
        "\t\t\tplot_harmonic_set(axes[:, j], param2,",
        "\t\t\t\t[(parameter_ranges[param2][q], by_point[(j, q)])",
        "\t\t\t\t\tfor q in range(0, range_size)],",
        "\t\t\t\ttitle=f'{param1}={parameter_ranges[param1][j]}',",
        "\t\t\t\tylabel='Current (A)' if j == 0 else '')",
        "\t\tfig.suptitle(f'{param1} vs {param2}')",
        "\t\tfinish_figure(fig, f'{param1}_vs_{param2}_harmonics')",
        "show_figures()",
    ]


def _figures():
    """`finish_figure` and `show_figures`, what every figure ends up in.

    Saving and closing as each figure is finished, rather than holding them all
    open for one `plt.show()` at the end, is what makes a long sweep survivable:
    a pairwise sweep over six parameters draws fifteen figures of
    `range_size` squared axes each, and matplotlib keeps every one of them alive
    until it is closed. It also means a sweep left running unattended leaves its
    results on disk rather than behind a window nobody was there to dismiss.

    Setting `figure_directory` to None in the script restores the old
    behaviour: nothing is written, the figures stay open, and `show_figures`
    blocks at the end of the sweep.
    """
    return [
        "def finish_figure(fig, name):",
        '\t"""Write one figure to `figure_directory` and close it."""',
        "\tif figure_directory is None:",
        "\t\treturn",
        "\tos.makedirs(figure_directory, exist_ok=True)",
        "\tfig.savefig(os.path.join(figure_directory, f'{name}.png'))",
        "\t#Closing is the point: an open figure holds its axes and data alive.",
        "\tplt.close(fig)",
        "",
        "",
        "def show_figures():",
        '\t"""Display the sweep\'s figures, unless they were saved and closed."""',
        "\tif figure_directory is None:",
        "\t\tplt.show()",
        "",
        "",
        "def plot_harmonic_set(axes, parameter, curves, **kwargs):",
        '\t"""Draw `harmonics` of each (value, current) in `curves` down `axes`."""',
        "\t#plot_harmonics takes one `<label>_data` keyword per trace, and labels",
        "\t#the trace with whatever comes before `_data`.",
        "\ttraces = {f'{parameter}={value}_data': {'time': x_vals,",
        "\t\t'current': current, 'harmonics': list(harmonics)}",
        "\t\tfor value, current in curves}",
        "\tsci.plot.plot_harmonics(axes_list=axes, xlabel=x_label,",
        "\t\tplot_func=harmonics_plot_func, **traces, **kwargs)",
        "\taxes[0].figure.tight_layout()",
    ]


def _runner():
    """`simulate_point` and `run_sweep`, what the sweep runs each point through.

    Both sit at the top level of the generated script so `pool.map` can pickle
    `simulate_point` by name; a sweep point, a dict of floats, and the current
    it returns are then all that crosses between processes. The experiment
    itself never does -- a compiled pydiffsol Ode does not pickle, which is why
    MechanismHandler refuses the multiprocessing dispersion path -- so each
    worker uses one of its own, inherited across the fork on Linux or built by
    re-running the setup above when the start method is spawn.

    The pool is made on first use and kept, rather than one per figure: under
    spawn every new worker rebuilds the model, so a pool per figure would pay
    that again for each of them.
    """
    return [
        "def simulate_point(overrides):",
        '\t"""Simulate one sweep point. `overrides` maps parameter name to value."""',
        "\texperiment.fixed_parameters = {**fixed_parameters, **overrides}",
        "\treturn experiment.dim_i(experiment.simulate([], times))",
        "",
        "",
        "pool = None",
        "",
        "",
        "def run_sweep(points):",
        '\t"""Simulate every point behind one figure, over `parallel_cpu` processes."""',
        "\tglobal pool",
        "\tif parallel_cpu == 1:",
        "\t\treturn [simulate_point(x) for x in points]",
        "\tif pool is None:",
        "\t\tpool = mp.Pool(processes=parallel_cpu)",
        "\t#chunksize=1 hands out one point at a time rather than in fixed",
        "\t#contiguous batches. Points cost wildly different amounts -- a high",
        "\t#Ru is far stiffer to solve than a low one -- so a fixed split can",
        "\t#strand the slow ones on one worker. Dispatching per point costs",
        "\t#microseconds against a solve measured in seconds. On a sweep whose",
        "\t#total is dominated by one very slow point it makes no odds either",
        "\t#way; it is the sweeps with many moderately uneven points it helps.",
        "\treturn pool.map(simulate_point, points, chunksize=1)",
    ]


def _guarded(lines):
    """Put `lines` under `if __name__ == "__main__":`.

    Workers started with the spawn start method import the script again, so
    anything that draws or sweeps has to sit behind the guard or every worker
    would do it too. The setup above the guard is left exposed on purpose: that
    is what rebuilds the experiment a spawned worker needs.
    """
    body = []
    for line in lines:
        body.extend("\t" + part if part else "" for part in line.split("\n"))
    return ['if __name__ == "__main__":'] + body


def parameter_scan_script(
    mechanism,
    experiment_type,
    path=None,
    mode="individual",
    potential_scan=True,
    range_size=4,
    potential_input=None,
    parallel_cpu=1,
    figure_directory=_FIGURE_DIRECTORY,
    harmonics=None,
    current_figsize=_CURRENT_FIGSIZE,
    harmonics_figsize=_HARMONICS_FIGSIZE,
):
    """Generate a script that sweeps every parameter a mechanism declares.

    Args:
        mechanism (str | os.PathLike | Mapping): YAML path, YAML document or
            parsed mapping -- anything the `mechanism` option accepts
        experiment_type (str): one of the SingleExperiment experiment types
        path (str, optional): write the script here. Omitted, nothing is
            written and the source is only returned.
        mode (str): "individual" sweeps one parameter per figure, "pairwise"
            crosses every pair on a grid
        potential_scan (bool): before the sweep, show a grid of potential
            waveforms -- one subplot per input parameter, each perturbed on its
            own by the scalars in `_INPUT_SCALARS`
        range_size (int): points per sweep
        potential_input: sympy expression for the potential, for
            experiment_type="Generic"
        parallel_cpu (int): worker processes the sweep is spread over. 1
            simulates in the script's own process; above that, the points
            behind each figure are mapped over a pool of this many. Written
            into the script as a variable, so it can be changed there without
            regenerating.
        figure_directory (str | os.PathLike | None): each figure is written
            here as a PNG named after what it sweeps, and closed as soon as it
            is saved. The directory is created by the script if it does not
            exist. None instead keeps every figure open and shows them at the
            end of the sweep. Written into the script as a variable, like
            `parallel_cpu`.
        harmonics (list[int] | False | None): harmonics drawn, with
            sci.plot.plot_harmonics, in a separate figure alongside each total
            current figure -- one column of them in individual mode, one column
            per value of the first parameter in pairwise. None plots 1 to 8 if
            the waveform has a sinusoidal component (an `omega` input) and none
            otherwise; False plots none. Written into the script as a variable,
            where None turns them off. They are plotted as absolute values,
            set by `harmonics_plot_func` in the script (np.real for the
            oscillation itself).
        current_figsize (tuple): (width, height) in inches of the total current
            figures. Written into the script as a variable.
        harmonics_figsize (tuple): (width, height) in inches of the harmonic
            figures. Written into the script as a variable.

    Returns:
        str: the generated source

    Raises:
        ValueError: for an unknown mode or experiment type, a `parallel_cpu`
            below one, a `figure_directory` that is neither a path nor None,
            harmonics that are not positive integers, a figure size that is not
            two positive numbers, or if the mechanism declares no parameter this
            knows how to sweep
    """
    if mode not in _MODES:
        raise ValueError(
            "`mode` keyword needs to be one of {0}, not {1!r}".format(_MODES, mode)
        )
    #bool is an int, and `parallel_cpu=True` asking for one worker is far more
    #likely a mistyped flag than a deliberate serial sweep.
    if not isinstance(parallel_cpu, int) or isinstance(parallel_cpu, bool):
        raise ValueError(
            "`parallel_cpu` keyword needs to be an integer, not {0!r}".format(
                parallel_cpu
            )
        )
    if parallel_cpu < 1:
        raise ValueError(
            "`parallel_cpu` keyword needs to be at least 1, not {0}".format(
                parallel_cpu
            )
        )
    if figure_directory is not None and not isinstance(
        figure_directory, (str, os.PathLike)
    ):
        raise ValueError(
            "`figure_directory` keyword needs to be a path, or None to show "
            "the figures instead of saving them, not {0!r}".format(figure_directory)
        )
    for name, size in (
        ("current_figsize", current_figsize),
        ("harmonics_figsize", harmonics_figsize),
    ):
        if not _positive_numbers(size) or len(size) != 2:
            raise ValueError(
                "`{0}` keyword needs to be (width, height) in inches, not "
                "{1!r}".format(name, size)
            )
    if harmonics is not None and harmonics is not False:
        if (
            not _positive_numbers(harmonics)
            or len(harmonics) == 0
            or not all(isinstance(x, (int, np.integer)) for x in harmonics)
        ):
            raise ValueError(
                "`harmonics` keyword needs to be a list of positive integers, "
                "None or False, not {0!r}".format(harmonics)
            )
        #Plain ints, so the list is written into the script as `[1, 2, 3]`
        #rather than numpy reprs.
        harmonics = [int(x) for x in harmonics]
    input_names = _input_names(experiment_type, potential_input)
    if harmonics is None:
        harmonics = (
            list(_DEFAULT_HARMONICS)
            if _oscillates(input_names, potential_input)
            else None
        )
    elif harmonics is False:
        harmonics = None
    #The cell parameters do not enter the waveform, so perturbing them would
    #draw the same curve three times over.
    waveform_names = [x for x in input_names if x not in _CELL_INPUTS]

    network = load_network(mechanism_source(mechanism))
    #The reaction each parameter belongs to. `reactions` is built from
    #`mech["reactions"]` in order, one entry each, so they zip; the equation is
    #only kept in the raw mechanism, since parse_reaction splits it up.
    reaction_of = {}
    for (reaction_name, reaction), entry in zip(
        network.reactions.items(), network.mech["reactions"]
    ):
        for name in reaction["params"]:
            reaction_of[name] = (reaction_name, entry["equation"])

    sweeps, fixed, headings = [], [], {}
    labelled = set()
    for name in required_parameters(network):
        expression, midpoint = _sweep(name, range_size)
        if expression is None:
            continue
        #Heads the group with the first of its parameters that survives the
        #sweep filter, rather than the first the reaction declares, so a
        #parameter family this does not know how to sweep cannot lose the label.
        #Keyed on the reaction name, so two steps written identically still get
        #a heading each.
        source, equation = reaction_of.get(name, (None, None))
        if source not in labelled:
            headings[name] = equation if equation is not None else "not reaction specific"
            labelled.add(source)
        sweeps.append((name, expression))
        fixed.append((name, repr(midpoint)))
    if not sweeps:
        raise ValueError(
            "mechanism declares no parameters that {0} knows how to sweep; "
            "known families are {1}".format(
                parameter_scan_script.__name__, sorted(_RANGE_VALUES)
            )
        )

    libraries = list(_LIBRARIES)
    if potential_input is not None:
        libraries.append(("sympy", "sympy"))
    imports = "\n".join(
        ["import {0} as {1}".format(module, alias) for module, alias in libraries]
        + ["import Surface_confined_inference as sci"]
    )
    construction = (
        'experiment = sci.SingleExperiment("{0}", input_parameter_values,\n'
        "\tmechanism={1}".format(experiment_type, _mechanism_literal(mechanism))
    )
    if potential_input is not None:
        construction += ",\n\tpotential_input=sympy.sympify({0!r})".format(
            str(potential_input)
        )
    construction += ")"

    setup = [
        "range_size = {0}".format(range_size),
        "",
        _dict_block("parameter_ranges", sweeps),
        "",
        _dict_block("fixed_parameters", fixed, headings=headings),
        "",
        _dict_block(
            "input_parameter_values",
            [(name, "None") for name in input_names],
            comment="  # TODO: set numeric values for the input potential waveform",
        ),
        "",
        "potential_scan = {0}".format(potential_scan),
        #Read by run_sweep, so raising it here is all it takes to spread an
        #already generated sweep over more cores.
        "parallel_cpu = {0}".format(parallel_cpu),
        #Read by finish_figure and show_figures; None there shows the figures
        #rather than writing them.
        "figure_directory = {0!r}".format(
            None if figure_directory is None else os.fspath(figure_directory)
        ),
        #Scaled, not offset, so one list covers parameters of wildly different
        #magnitude. The cost is that a parameter sitting at zero (Estart on a
        #sweep starting from 0, phase on an unshifted FTACV) cannot move -- edit
        #the list here if one of those is the parameter of interest.
        #Read by the sweep loop and plot_harmonic_set. None skips the harmonic
        #figures altogether, which is what a waveform with no sinusoid gets.
        "harmonics = {0!r}".format(harmonics),
        #The envelope of each harmonic; np.real draws the oscillation itself.
        "harmonics_plot_func = np.abs",
        "current_figsize = {0!r}".format(tuple(current_figsize)),
        "harmonics_figsize = {0!r}".format(tuple(harmonics_figsize)),
        "input_scan_scalars = {0}".format(list(_INPUT_SCALARS)),
        "scanned_inputs = {0}".format(waveform_names),
        "",
        construction,
        #get_voltage resolves the waveform parameters through the parameter
        #interface, which is only built once optim_list has been set -- so both
        #have to be assigned before the potential can be asked for, even though
        #the sweep below reassigns them per parameter.
        "experiment.fixed_parameters = fixed_parameters",
        "experiment.optim_list = []",
        "times = experiment.calculate_times(dimensional=False)",
        "x_vals = experiment.dim_t(times)",
        "x_label = 'Time (s)'",
    ]

    source = (
        "\n".join(
            [imports, ""]
            + setup
            + ["", ""]
            + _figures()
            + ["", ""]
            + _runner()
            + ["", ""]
            + _guarded(_input_scan() + [""] + _loop(mode))
        )
        + "\n"
    )
    if path is not None:
        with open(path, "w") as f:
            f.write(source)
    return source
