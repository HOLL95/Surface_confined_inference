
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
)


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
        "\tplt.show()",
    ]


def _loop(mode):
    """The sweep itself.

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
            "\tfig, ax = plt.subplots()",
            "\tfor j in range(0, range_size):",
            "\t\tvalue = parameter_ranges[parameter][j]",
            "\t\texperiment.fixed_parameters = {**fixed_parameters, parameter: value}",
            "\t\tcurrent = experiment.dim_i(experiment.simulate([], times))",
            "\t\tax.plot(x_vals, current, label=f'{parameter}={value}')",
            "\tax.set_xlabel(x_label)",
            "\tax.set_ylabel('Current (A)')",
            "\tax.legend()",
            "plt.show()",
        ]
    return [
        "combinations = list(it.combinations(parameter_ranges.keys(), 2))",
        "for param1, param2 in combinations:",
        "\tfig, ax = plt.subplots(range_size, range_size)",
        "\tfor j in range(0, range_size):",
        "\t\tvalue1 = parameter_ranges[param1][j]",
        "\t\tfor q in range(0, range_size):",
        "\t\t\tvalue2 = parameter_ranges[param2][q]",
        "\t\t\texperiment.fixed_parameters = {**fixed_parameters,",
        "\t\t\t\tparam1: value1, param2: value2}",
        "\t\t\tcurrent = experiment.dim_i(experiment.simulate([], times))",
        "\t\t\tax[j, q].plot(x_vals, current, label=f'{param2}={value2}')",
        "\t\tax[j, 0].set_ylabel(f'{param1}={value1}')",
        "\tfor q in range(0, range_size):",
        "\t\tax[-1, q].set_xlabel(x_label)",
        "\tfig.suptitle(f'{param1} vs {param2}')",
        "\tplt.tight_layout()",
        "plt.show()",
    ]


def parameter_scan_script(
    mechanism,
    experiment_type,
    path=None,
    mode="individual",
    potential_scan=True,
    range_size=4,
    potential_input=None,
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

    Returns:
        str: the generated source

    Raises:
        ValueError: for an unknown mode or experiment type, or if the mechanism
            declares no parameter this knows how to sweep
    """
    if mode not in _MODES:
        raise ValueError(
            "`mode` keyword needs to be one of {0}, not {1!r}".format(_MODES, mode)
        )
    input_names = _input_names(experiment_type, potential_input)
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
        #Scaled, not offset, so one list covers parameters of wildly different
        #magnitude. The cost is that a parameter sitting at zero (Estart on a
        #sweep starting from 0, phase on an unshifted FTACV) cannot move -- edit
        #the list here if one of those is the parameter of interest.
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
            [imports, ""] + setup + [""] + _input_scan() + [""] + _loop(mode)
        )
        + "\n"
    )
    if path is not None:
        with open(path, "w") as f:
            f.write(source)
    return source
