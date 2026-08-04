"""Pipeline stages for the YAML mechanism model.

The stages run in order, each consuming the previous one's product:

    load_network -> build_odes -> render_potential -> render_diffsl -> build_solver

In `yml_constructor_class.MechanismModel` these stages communicated through
instance attributes (`self.species_map`, `self.params`, `self.overpotential_range`,
`self.poly_degree`, `self.ode`), which made the call order load-bearing but
unenforced -- `simulate` needed three earlier methods to have run, in a particular
order, or it failed with a bare AttributeError. Here each stage returns its product
and the next stage takes it as an argument, so the ordering is structural.

Pure value-in/value-out transforms live in `_MechanismHelpers`; this module is the
only one that touches YAML, the solver, or accumulated state.
"""
import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pydiffsol as ds
import sympy
import yaml

from ._MechanismHelpers import (
    PARAM_DICT,
    electron_count,
    even_coefficient_count,
    heaviside_switchpoints,
    marcus_coefficients,
    parse_reaction,
    sympy_to_diffsl,
    validate_charge_position,
    waveform,
)

#Every DiffSL input is declared as "<name>=0"; the suffix is how a declaration
#string and a parameter name convert into one another.
_DECLARATION_SUFFIX = "=0"


def _declare(name):
    return "{0}{1}".format(name, _DECLARATION_SUFFIX)


"""
Stage products
"""


@dataclass(frozen=True)
class ReactionNetwork:
    """Species and reactions parsed out of a mechanism definition.

    Attributes:
        species (dict): species name -> location ("surface", ...)
        reactions (dict): reaction name -> reactants/products/type/params
        mech (dict): the raw parsed mechanism, kept for metadata
    """

    species: dict
    reactions: dict
    mech: dict


@dataclass(frozen=True)
class OdeSystem:
    """Symbolic ODEs for the independent species.

    Attributes:
        equations (dict): state -> d(state)/dt expression
        electron_terms (dict): {"ox": [...], "red": [...]} rate terms
        species_map (dict): original species name -> DiffSL-safe name
    """

    equations: dict
    electron_terms: dict
    species_map: dict


@dataclass(frozen=True)
class PotentialBlock:
    """Rendered DiffSL for the input potential and its derivative.

    Attributes:
        E (str): potential tensor declaration
        dE (str): derivative tensor declaration
        parameters (list): declarations for any symbols left free in the waveform
        stepped (bool): True when the waveform is gated and needs stop_i/reset_i
        stop_block (str): stop_i declaration, or None when not stepped
    """

    E: str
    dE: str
    parameters: list
    stepped: bool
    stop_block: str


@dataclass(frozen=True, eq=False)
class ParameterLayout:
    """Ordered DiffSL input slots, addressable by parameter name.

    The code this replaces indexed the input vector by rebuilding declaration
    strings at pack time (`name.replace("_", "") + "=0"`). Two consequences: the
    final `elif param in self.params` branch compared a bare name against
    "name=0" strings and so could never fire, which meant a mistyped parameter
    was silently simulated as zero; and the ordering contract lived in string
    surgery rather than anywhere nameable.

    Attributes:
        declarations (tuple): input declarations, in DiffSL order
        poly_degree (int): coefficients per rate polynomial
        species_initials (tuple): (declaration, value) for the initial coverages
        overpotential_range (array): window the Marcus fit uses; attached by
            build_solver, since it depends on the potential actually reached
    """

    declarations: tuple
    poly_degree: int
    species_initials: tuple
    overpotential_range: object = None

    @property
    def names(self):
        """Declared parameter names, in DiffSL order."""
        return tuple(x[: -len(_DECLARATION_SUFFIX)] for x in self.declarations)

    def with_initials(self, initials):
        """Return a copy whose initial state is `initials` ({name: value}).

        The default is an equal split of coverage across the species with the
        current starting at zero. That is not the only sensible convention --
        Surface_confined_inference starts fully reduced with the capacitive
        current already flowing -- so the choice belongs to the caller.
        """
        return replace(
            self,
            species_initials=tuple(
                (self._slot("{0}0".format(name)), initials[name]) for name in initials
            ),
        )

    def _slot(self, name, requested=None):
        """Declaration for `name`, reported against `requested` as the caller spelt it."""
        declaration = _declare(name)
        if declaration not in self.declarations:
            raise KeyError(
                "{0!r} is not declared by this model".format(
                    name if requested is None else requested
                )
            )
        return declaration

    def pack(self, parameters):
        """Build the DiffSL input vector from named parameter values.

        Args:
            parameters (dict): parameter name -> value. `k0_<i>` is expanded into
                the fitted rate polynomial for reaction i, consuming `lambda_<i>`.

        Returns:
            numpy.ndarray: values in declaration order

        Raises:
            KeyError: if a name is not declared by this model, or if a `k0_<i>`
                is supplied without its `lambda_<i>`
        """
        slots = dict.fromkeys(self.declarations, 0.0)
        for name in parameters:
            if "E0" in name or "kcat" in name:
                #E0_1 is declared as E01: the index is a suffix, not a separate name
                slots[self._slot(name.replace("_", ""), name)] = parameters[name]
            elif "k0" in name:
                index = name.split("_")[-1]
                lambda_name = "lambda_" + index
                if lambda_name not in parameters:
                    raise KeyError(
                        "{0!r} needs {1!r} to fit its rate polynomial".format(
                            name, lambda_name
                        )
                    )
                if self.overpotential_range is None:
                    raise RuntimeError(
                        "Rate polynomials need an overpotential window; pack the "
                        "layout returned by build_solver, not the one from render_diffsl"
                    )
                gcoeffs = marcus_coefficients(
                    parameters[lambda_name],
                    parameters[name],
                    self.overpotential_range,
                    poly_degree=self.poly_degree,
                )
                for order in range(0, len(gcoeffs)):
                    slots[self._slot("g{0}{1}".format(index, order))] = gcoeffs[order]
            elif name.startswith("lambda"):
                #Consumed by the k0 expansion above; it has no slot of its own.
                continue
            else:
                slots[self._slot(name)] = parameters[name]
        #Applied after the loop, so an explicitly supplied coverage does not win
        #over the equal split -- matching the order the original packed in.
        for declaration, value in self.species_initials:
            slots[declaration] = value
        return np.array([slots[x] for x in self.declarations])


@dataclass(frozen=True, eq=False)
class DiffslModel:
    """Generated DiffSL source together with the layout of its inputs.

    Attributes:
        source (str): the DiffSL
        layout (ParameterLayout): ordered input slots
        state_names (tuple): integrated states, in row order -- the independent
            species followed by the current. The species eliminated by the
            conservation sum is not among them.
    """

    source: str
    layout: ParameterLayout
    state_names: tuple


@dataclass(frozen=True, eq=False)
class CompiledMechanism:
    """A compiled solver plus everything needed to drive it.

    Attributes:
        ode: compiled pydiffsol Ode
        layout (ParameterLayout): input layout, with the overpotential window set
        potential_start (float): potential at the first requested time
        source (str): the DiffSL the solver was built from
    """

    ode: object
    layout: ParameterLayout
    potential_start: float
    source: str
    state_names: tuple

    def solve(self, parameters, times):
        """Solve for `times`, packing `parameters` into the input vector."""
        return self.ode.solve_dense(self.layout.pack(parameters), times)


"""
Stage 1 -- the reaction network
"""


def mechanism_source(source):
    """Resolve a mechanism definition to a plain mapping.

    Accepts a path to a YAML file, a YAML document as a string, or an
    already-parsed mapping. The mapping is what should be persisted: a path
    does not travel between machines, so anything saved and reloaded elsewhere
    (`save_class`, the slurm flow) has to carry the parsed content.

    Args:
        source (str | os.PathLike | Mapping)

    Returns:
        dict

    Raises:
        TypeError: if `source` is not a path, YAML string or mapping
        ValueError: if a YAML string does not parse to a mapping
    """
    if source is None:
        raise ValueError(
            "the `mechanism` option must be a YAML path, a YAML string or a "
            "parsed mapping, not None"
        )
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    if isinstance(source, (str, os.PathLike)):
        #A path wins over parsing the value as a document, so a filename that
        #happens to be valid YAML still reads the file it names.
        if os.path.isfile(source):
            with open(source) as f:
                return yaml.safe_load(f)
        parsed = yaml.safe_load(source)
        if not isinstance(parsed, Mapping):
            raise ValueError(
                "mechanism {0!r} is neither an existing file nor a YAML mapping".format(
                    source
                )
            )
        return dict(parsed)
    raise TypeError(
        "mechanism must be a path, a YAML string or a mapping, not {0}".format(
            type(source).__name__
        )
    )


def required_parameters(network):
    """Parameter names this mechanism needs supplied.

    Every reaction contributes the parameters its type declares (`k0_1`,
    `lambda_1`, `E0_1` for a Marcus step; `kcatf_1`/`kcatb_1` for a catalytic
    one), and every mechanism needs the three cell parameters.

    Args:
        network (ReactionNetwork)

    Returns:
        list: parameter names, reaction parameters first
    """
    names = []
    for reaction in network.reactions.values():
        names += reaction["params"]
    return names + ["gamma", "Cdl", "Ru"]


def load_network(source):
    """Parse a mechanism into species and reactions.

    Args:
        source (str | os.PathLike | Mapping): anything `mechanism_source` takes

    Returns:
        ReactionNetwork

    Raises:
        ValueError: if a reaction names a species the file does not declare
    """
    mech = mechanism_source(source)

    counter = {key: 1 for key in PARAM_DICT.keys()}
    species = dict(
        zip(
            [x["name"] for x in mech["species"]],
            [x["location"] for x in mech["species"]],
        )
    )
    reactions = {}
    for entry in mech["reactions"]:
        elems = parse_reaction(entry["equation"])
        reaction_type = entry["type"]
        reaction_name = "{0}_{1}".format(reaction_type, counter[reaction_type])

        reactions[reaction_name] = elems
        reactions[reaction_name]["params"] = [
            "{0}_{1}".format(x, counter[reaction_type])
            for x in PARAM_DICT[reaction_type]
        ]
        counter[reaction_type] += 1
        for key in ["products", "reactants"]:
            for name in elems[key]:
                if name not in species:
                    raise ValueError(
                        "Yaml mechanism file does not contain species {0}, "
                        "(from reaction {1})".format(name, entry["equation"])
                    )
    return ReactionNetwork(species=species, reactions=reactions, mech=mech)


"""
Stage 2 -- symbolic ODEs
"""


def build_odes(network):
    """Build the ODE for each independent species.

    One surface species is eliminated by the conservation sum (its coverage is
    implied by the others), so it gets no state and no equation.

    Args:
        network (ReactionNetwork)

    Returns:
        OdeSystem

    Raises:
        NotImplementedError: if the mechanism declares no surface species
    """
    species_dict, reactions = network.species, network.reactions
    species = list(species_dict.keys())
    sympy_species = {key: sympy.symbols(key) for key in species}
    replacement = None
    for i in range(0, len(species)):
        if species_dict[species[i]] == "surface":
            replacement = species[i]
            break
    if replacement is None:
        raise NotImplementedError
    substitution = 1
    for i in range(0, len(species)):
        if species_dict[species[i]] == "surface" and species[i] != replacement:
            substitution -= sympy_species[species[i]]

    sympy_species[replacement] = substitution
    independent = [x for x in species if x != replacement]
    equation_dict = {sympy.symbols(key): sympy.Integer(0) for key in independent}
    electron_dict = {"ox": [], "red": []}
    reaction_list = list(reactions.keys())
    for i in range(0, len(reaction_list)):
        r = reaction_list[i]
        reaction_num = int(r.split("_")[-1])
        nu = {x: 0 for x in species}
        for x in reactions[r]["reactants"]:
            nu[x] -= 1
        for x in reactions[r]["products"]:
            nu[x] += 1
        products = sympy.prod([sympy_species[x] for x in reactions[r]["products"]])
        reactants = sympy.prod([sympy_species[x] for x in reactions[r]["reactants"]])
        if "marcus" in r:
            kox = sympy.symbols("kox{0}".format(reaction_num))
            kred = sympy.symbols("kred{0}".format(reaction_num))
            if len(reactions[r]["products"]) == 1 and len(reactions[r]["reactants"]) == 1:
                couple = [reactions[r]["reactants"][0], reactions[r]["products"][0]]
                [validate_charge_position(x) for x in couple]
                _, charges = electron_count(couple)
                if charges[0] < charges[1]:  # reactant loses electron, is being oxidised
                    reactants *= kox
                    products *= kred
                    iterable = zip([reactants, products], ["ox", "red"])
                else:  # reactant gains an electron, is being reduced
                    reactants *= kred
                    products *= kox
                    iterable = zip([reactants, products], ["red", "ox"])
                for elem, key in iterable:
                    electron_dict[key].append(elem)
        else:
            kf = sympy.symbols("kcatf{0}".format(reaction_num))
            kb = sympy.symbols("kcatb{0}".format(reaction_num))
            reactants *= kf
            products *= kb
        net = reactants - products if reactions[r]["type"] == "reversible" else reactants
        for x in independent:
            if nu[x] != 0:
                equation_dict[sympy.symbols(x)] += nu[x] * net

    #A trailing charge is not a legal DiffSL identifier, so rename O- to Ominus
    #and rewrite every reference to it.
    offender = ["+", "-"]
    replacer = ["plus", "minus"]
    ns = {}
    species_map = {}
    for symbol in equation_dict.keys():
        str_symbol = str(symbol)
        new_symbol = None
        for off, rep in zip(offender, replacer):
            if str_symbol[-1] == off:
                new_symbol = str_symbol[:-1] + rep
        if new_symbol is not None:
            ns[symbol] = new_symbol
            for symbol2 in equation_dict.keys():
                equation_dict[symbol2] = equation_dict[symbol2].subs(symbol, new_symbol)
                for reac in ["ox", "red"]:
                    electron_dict[reac] = [
                        x.subs(symbol, new_symbol) for x in electron_dict[reac]
                    ]
    for old_symbol in ns.keys():
        equation_dict[ns[old_symbol]] = equation_dict[old_symbol]
        species_map[str(old_symbol)] = ns[old_symbol]
        del equation_dict[old_symbol]
    return OdeSystem(
        equations=equation_dict, electron_terms=electron_dict, species_map=species_map
    )


"""
Stage 3 -- the input potential
"""


def render_potential(expr, time_var="t"):
    """Render the potential and its derivative as DiffSL tensors.

    A waveform built from Heaviside gates cannot be differentiated pointwise, so
    it is rendered as one piece per interval between switchpoints and indexed by
    the solver's stop-root counter.

    Args:
        expr: sympy expression for the potential
        time_var (str): name of the time symbol

    Returns:
        PotentialBlock
    """
    deriv_term = sympy.symbols(time_var)
    E_params = []
    for x in expr.free_symbols:
        strx = str(x)
        if strx != time_var:
            E_params.append(_declare(strx))
    if not expr.has(sympy.Heaviside):
        dE = sympy.diff(expr, deriv_term)
        return PotentialBlock(
            E="E {%s}\n" % sympy_to_diffsl(expr),
            dE="dE {%s}\n" % sympy_to_diffsl(dE),
            parameters=E_params,
            stepped=False,
            stop_block=None,
        )

    switchpoints = np.array(heaviside_switchpoints(expr, deriv_term))
    positive_switchpoints = [0] + list(switchpoints[switchpoints > 0])
    # N is the index of the stop root that fired most recently, initialised
    # to 0 -- it is not a count of events. Dropping the leading switchpoint
    # made the first two pieces both read E_i[0] and shifted every later
    # piece by one, so keep t=0 in the list: root j then selects piece j.
    # The t-0 root sits on the initial time and never fires, and N=0 is
    # already the piece that is wanted before any switch, so the alignment
    # holds either way.
    time_params = (
        "stop_i {"
        + ",".join(["t-{0}".format(x) for x in positive_switchpoints])
        + "}"
    )

    E_arr = []
    dE_arr = []
    for i in range(0, len(positive_switchpoints)):
        if i == len(positive_switchpoints) - 1:
            hi = positive_switchpoints[i] + 1
        else:
            hi = positive_switchpoints[i + 1]
        mid = (positive_switchpoints[i] + hi) / 2
        expr_on_piece = expr
        for h in expr.atoms(sympy.Heaviside):
            expr_on_piece = expr_on_piece.subs(h, h.subs(deriv_term, mid))
        simplified = sympy.simplify(expr_on_piece)
        E_arr.append(sympy_to_diffsl(simplified))
        dE_arr.append(sympy_to_diffsl(sympy.diff(simplified, deriv_term)))
    return PotentialBlock(
        E="E_i {%s}\n" % (",".join(E_arr)),
        dE="dE_i {%s}\n" % (",".join(dE_arr)),
        parameters=E_params,
        stepped=True,
        stop_block=time_params,
    )


"""
Stage 4 -- DiffSL source
"""


def render_diffsl(network, odes, potential, poly_degree=15):
    """Emit the DiffSL model and the layout of its inputs.

    Args:
        network (ReactionNetwork)
        odes (OdeSystem)
        potential (PotentialBlock)
        poly_degree (int): coefficients per rate polynomial. Butler-Volmer is
            exact at 2; high degrees are not free even where the extra
            coefficients are zero, since the dead pow() terms destabilise the
            implicit solvers.

    Returns:
        DiffslModel
    """
    reactions = network.reactions
    redox_dict = {}
    n_even = even_coefficient_count(poly_degree)
    num_redox = 0
    num_cat = 0
    for reaction in reactions:
        if "marcus" in reaction:
            index = reaction.split("_")[-1]
            #g is even in the overpotential and shared by both directions, so one
            #polynomial in u = op^2 replaces the two full polynomials in op that
            #used to be fitted independently -- half the coefficients, and the
            #exact +-op/2 split is applied afterwards rather than fitted.
            redox_dict["u{0}".format(index)] = "op{0}*op{0}".format(index)
            #Horner rather than a sum of pow(): identical arithmetic, but a
            #multiply-add chain instead of one pow() call per order.
            expression = "g{0}{1}".format(index, n_even - 1)
            for k in range(n_even - 2, -1, -1):
                expression = "({0})*u{1} + g{1}{2}".format(expression, index, k)
            redox_dict["g{0}".format(index)] = expression
            redox_dict["kox{0}".format(index)] = "exp(g{0} + 0.5*op{0})".format(index)
            redox_dict["kred{0}".format(index)] = "exp(g{0} - 0.5*op{0})".format(index)
            num_redox += 1
        else:
            num_cat += 1
    #Counts become exclusive upper bounds for the 1-based reaction indices below.
    if num_redox > 0:
        num_redox += 1
    if num_cat > 0:
        num_cat += 1

    Faradaic_equations, electron_dict = odes.equations, odes.electron_terms
    state_names = [str(x) for x in list(Faradaic_equations.keys()) + ["I"]]
    # named form "R = R0": the anonymous form "u_i {R0}" takes the initial value
    # from the input but leaves the state unnamed, so F_i/op/dtheta cannot refer
    # to it (and a name that happens to collide resolves to the constant input).
    inits = ["{0} = {0}0".format(name) for name in state_names]
    init_params = ["{0}0=0".format(name) for name in state_names]

    u_i = "u_i {%s}" % (",\n".join(inits))

    dtheta = (
        "{-("
        + " + ".join([str(x) for x in electron_dict["red"]])
        + ") + ("
        + " + ".join([str(x) for x in electron_dict["ox"]])
        + ")}"
    )

    if potential.stepped == False:
        ops = "\n".join(["op%s {E-E0%s-I*Ru}" % (x, x) for x in range(1, num_redox)])
        dIdt = "-(1/(Ru*Cdl))*(I-gamma*dtheta-Cdl*dE)"
    else:
        ops = "\n".join(["op%s {E_i[N]-E0%s-I*Ru}" % (x, x) for x in range(1, num_redox)])
        dIdt = "-(1/(Ru*Cdl))*(I-gamma*dtheta-Cdl*dE_i[N])"
    ks = "\n".join(["%s {%s}" % (key, redox_dict[key]) for key in redox_dict.keys()])
    F_i = "F_i {%s, %s}" % (
        ",".join([sympy_to_diffsl(Faradaic_equations[x]) for x in Faradaic_equations.keys()]),
        dIdt,
    )
    # gamma*dtheta is the Faradaic current, so it is reported, not integrated. It
    # used to be a third state with dy/dt = gamma*dtheta, which made the row
    # `Faradaic_only` reads the accumulated charge rather than the current.
    # out_i evaluates it at each output point without adding a state.
    out_i = "out_i {%s, gamma*dtheta}" % ",".join(state_names)
    redox_params = [
        "g{0}{1}=0".format(i, k)
        for i in range(1, num_redox)
        for k in range(0, n_even)
    ]
    e0_params = ["E0{0}=0".format(index) for index in range(1, num_redox)]
    cat_params = [
        "kcat{0}{1}=0".format(r, index)
        for r in ("f", "b")
        for index in range(1, num_cat)
    ]
    E_params = potential.parameters
    params = (
        "in_i {Cdl=0, Ru=0, gamma=0, "
        + " , ".join(redox_params + e0_params + cat_params + E_params + init_params)
        + "}"
    )
    declarations = (
        [_declare(x) for x in ["Cdl", "Ru", "gamma"]]
        + redox_params
        + e0_params
        + cat_params
        + E_params
        + init_params
    )
    E, dE = potential.E, potential.dE
    # DiffSL resolves names in declaration order, so u_i must precede every
    # tensor that reads a state (op/k/dtheta/F). With u_i last, `I` and the
    # species names instead bound to the same-named in_i constants and the
    # model silently integrated with frozen coverage and no IR drop.
    return_str = f"""
            {params}\n
            {u_i}\n
            {E}
            {dE}
            {ops}\n
            {ks}\n
            dtheta {dtheta}\n
            {F_i}\n
            {out_i}\n
            """
    if potential.stepped == True:
        # stop_i only halts the solve at each switchpoint; without a reset the
        # solver returns the (truncated) solution up to the first root. reset_i
        # must have the same shape as u_i and gives the state to restart from,
        # so the identity reset continues the trajectory while still letting the
        # integrator re-initialise across the discontinuity in dE/dt.
        reset = "reset_i {%s}\n" % ",".join(state_names)
        source = return_str + potential.stop_block + "\n" + reset

    else:
        source = return_str

    share = 1 / len(network.species)
    species_initials = tuple(
        (_declare("{0}0".format(odes.species_map.get(name, name))), share)
        for name in network.species
        #The eliminated species is not a state; its coverage is implied by the
        #conservation sum, so it has no slot to write to.
        if _declare("{0}0".format(odes.species_map.get(name, name))) in declarations
    )
    layout = ParameterLayout(
        declarations=tuple(declarations),
        poly_degree=poly_degree,
        species_initials=species_initials,
    )
    return DiffslModel(source=source, layout=layout, state_names=tuple(state_names))


"""
Stage 5 -- the compiled solver
"""


def build_solver(
    model,
    times,
    input_potential,
    input_parameters,
    time_var="t",
    max_current=1e-4,
    max_resistance=1e3,
    ode_solver=None,
):
    """Compile the DiffSL and attach the overpotential window the fit needs.

    Args:
        model (DiffslModel): source and layout from render_diffsl
        times (array): times the model will be solved over
        input_potential: sympy expression for the potential, unsubstituted
        input_parameters (dict): values for the waveform's symbols
        max_current, max_resistance (float): bound the IR drop, which widens the
            overpotential window beyond the +-(max E - min E) the E0 shift can
            already reach
        ode_solver: pydiffsol solver type. Defaults to the explicit tsit45; the
            implicit options fail in the Newton solve when Ru*Cdl is far below
            the drive period. NB the original named this `tsit4`, which no
            longer exists in pydiffsol -- construct_simulator could not compile
            anything against the installed version.

    Returns:
        CompiledMechanism
    """
    t_sym = sympy.symbols(time_var)
    iparams = list(input_parameters.keys())
    param_syms = sympy.symbols(iparams)
    # lambdify once, over (t, *params), so it can be reused for every parameter
    # combination instead of re-substituting and re-lambdifying each time
    E_func = sympy.lambdify((t_sym, *param_syms), input_potential, modules="numpy")

    max_drop = max_current * max_resistance
    potential_values = E_func(times, *[input_parameters[p] for p in iparams])
    # The rate polynomials are evaluated at op_j = E - E0_j - I*Ru, so the window
    # has to span the E0 shift as well as the IR drop. Every formal potential
    # lies inside the swept range, so E0 needs no separate bound: the widest op
    # any couple can reach is +-(max(E) - min(E)). Fitting over E alone left
    # every couple extrapolating -- at E0 = -0.5 V (-19.47 RT/F) the evaluation
    # point ran ~19.5 units past the fit range, where the degree-14 polynomial
    # returns log k ~ 1.8e5 against a true ~5.
    span = max(potential_values) - min(potential_values)
    window = np.linspace(-span - max_drop, span + max_drop, 1000)
    ode = ds.Ode(
        model.source,
        matrix_type=ds.nalgebra_dense,
        ode_solver=ds.OdeSolverType.tsit45 if ode_solver is None else ode_solver,
    )
    return CompiledMechanism(
        ode=ode,
        layout=replace(model.layout, overpotential_range=window),
        potential_start=potential_values[0],
        source=model.source,
        state_names=model.state_names,
    )


"""
Whole pipeline
"""


def build(source, input_potential, input_parameters, times, poly_degree=15, time_var="t", **kwargs):
    """Run every stage: mechanism definition in, compiled solver out.

    Args:
        source: YAML path or parsed mapping, as load_network takes
        input_potential (str | sympy expression): a pre-made waveform name, or
            an expression in `time_var`
        input_parameters (dict): values for the waveform's symbols
        times (array): times the model will be solved over
        poly_degree (int): coefficients per rate polynomial
        **kwargs: forwarded to build_solver

    Returns:
        CompiledMechanism
    """
    if isinstance(input_potential, str):
        input_potential = waveform(input_potential, params=input_parameters)

    network = load_network(source)
    odes = build_odes(network)
    potential = render_potential(input_potential.subs(input_parameters), time_var)
    model = render_diffsl(network, odes, potential, poly_degree=poly_degree)
    return build_solver(
        model,
        times,
        input_potential,
        input_parameters,
        time_var=time_var,
        **kwargs,
    )
