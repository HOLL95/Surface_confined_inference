"""Pure helpers for the YAML mechanism model.

Every function here is a value-in/value-out transform: no instance state, no file
or console I/O, no mutation of its arguments. That is the whole basis of the split
from `_MechanismProcess` -- anything that accumulates state across calls, or that
only makes sense at a particular point in the build pipeline, belongs there.

Extracted from `yml_constructor_class.MechanismModel` without behavioural change.
The comments explaining *why* each piece is shaped the way it is have been kept
with the code they explain.

Requires Python 3.9+ for `ast.unparse` (as did the original).
"""
import ast
import re

import numpy as np
import sympy
from scipy.integrate import quad

#Parameters each reaction type contributes, in the order they are numbered.
PARAM_DICT = {
    "marcus": ["k0", "lambda", "E0"],
    "catalytic": ["kcatf", "kcatb"],
    "BV": ["k0", "E0"],
}

_ARROWS = {"<->": "reversible", "<-": "irreversible", "->": "irreversible"}
_SIGN = {"+": 1, "-": -1}
_CHARGE_PATTERN = re.compile(r"(\d?[+-])$")

#DiffSL spells the inverse hyperbolics differently to sympy.
_FUNCTION_NAMES = {
    "asinh": "arcsinh",
    "acosh": "arcscosh",
}


"""
Reaction string parsing
"""


def split_species(string):
    """Split one side of a reaction equation into species names."""
    return [x for x in string.split(" ") if (x != "+" and x != "")]


def parse_reaction(string):
    """Parse `A + B <-> C` into reactants, products and reversibility.

    Args:
        string (str): reaction equation containing one of `<->`, `->`, `<-`

    Returns:
        dict: {"reactants": [...], "products": [...], "type": reversible|irreversible}

    Raises:
        ValueError: if the equation contains no reaction arrow
    """
    for arrow in ("<->", "->"):
        if arrow in string:
            parts = string.split(arrow)
            return {
                "products": split_species(parts[1]),
                "reactants": split_species(parts[0]),
                "type": _ARROWS[arrow],
            }
    if "<-" in string:
        parts = string.split("<-")
        #Reversed relative to the forward arrows: the products are written first.
        return {
            "products": split_species(parts[0]),
            "reactants": split_species(parts[1]),
            "type": _ARROWS["<-"],
        }
    raise ValueError("Reaction {0} has no reaction arrow (<->, ->, <-)".format(string))


def validate_charge_position(species):
    """Reject a species whose charge symbol is anywhere but the last character.

    Raises:
        ValueError: if `+` or `-` appears other than at the end
    """
    last = len(species) - 1
    for elem in ("+", "-"):
        if elem in species and any(
            (species[x] == elem and x != last) for x in range(0, len(species))
        ):
            raise ValueError(
                "Species ({0}) needs to have charge {1} at the end, nowhere else".format(
                    species, elem
                )
            )


def formal_charge(species):
    """Signed charge parsed off the end of a species name (`O2-` -> -2, `O` -> 0)."""
    match = _CHARGE_PATTERN.search(species)
    charge = match.group(1) if match else 0
    if charge == 0:
        return 0
    if len(charge) == 1:
        return _SIGN[charge]
    return int(charge[0]) * _SIGN[charge[1]]


def electron_count(couple):
    """Net electrons transferred across a redox couple [reactant, product].

    Raises:
        ValueError: if the couple transfers no electrons
        NotImplementedError: for multi-electron steps
    """
    charges = [formal_charge(x) for x in couple]
    n = max(charges) - min(charges)
    if n == 0:
        raise ValueError(
            "Electron transfer reaction {0}<->{1} has a net electron flow of 0".format(
                couple[0], couple[1]
            )
        )
    if n > 1:
        raise NotImplementedError
    return n, charges


"""
Symbolic expression -> DiffSL source
"""


class PowTransformer(ast.NodeTransformer):
    """Rewrite `a ** b` as `pow(a, b)`; DiffSL has no exponentiation operator."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            return ast.Call(
                func=ast.Name(id="pow", ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        return node


class TanhTransformer(ast.NodeTransformer):
    """Rewrite `tanh(x)` as `2*sigmoid(2*x) - 1`, which cannot overflow.

    DiffSL evaluates tanh through exp, so it returns NaN for |x| > log(DBL_MAX)
    = 709.78 instead of saturating at +/-1. That is not an edge case here: the
    smoothed staircase is one flat sum over every step, so every gate is
    evaluated at every time point, and the argument of the gate belonging to the
    last step is -(total scan time)/eps at t=0. A 20-step scan at smoothing=0.05
    emits tanh(40*t - 800), which is NaN for the whole of t < 2.26 -- the entire
    right-hand side goes NaN, the solver rejects every step it tries and shrinks
    the step size forever, and the solve never returns. The failure scales the
    wrong way, biting whenever N_steps > 355*smoothing, so a longer scan needs
    *more* smoothing to stay finite.

    sigmoid saturates cleanly at both ends (exp(-x) overflowing to inf gives
    1/inf = 0), so the identity tanh(x) = 2*sigmoid(2x) - 1 is exact where tanh
    is finite and gives +/-1 where it would have been NaN. Measured agreement
    with numpy.tanh over |x| in [0, 1e4] is 2.2e-16.
    """

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "tanh":
            doubled = ast.BinOp(
                left=ast.Constant(2.0), op=ast.Mult(), right=node.args[0]
            )
            sigmoid = ast.Call(
                func=ast.Name(id="sigmoid", ctx=ast.Load()),
                args=[doubled],
                keywords=[],
            )
            return ast.BinOp(
                left=ast.BinOp(left=ast.Constant(2.0), op=ast.Mult(), right=sigmoid),
                op=ast.Sub(),
                right=ast.Constant(1.0),
            )
        return node


def pow_to_call(source):
    """Return `source` with every `**` rewritten to a `pow()` call."""
    tree = ast.parse(source)
    tree = PowTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def sympy_to_diffsl(expr):
    """Render a sympy expression as DiffSL source."""
    text = str(expr)
    for key in _FUNCTION_NAMES:
        if key in text:
            text = text.replace(key, _FUNCTION_NAMES[key])
    tree = ast.parse(text)
    for transformer in (PowTransformer(), TanhTransformer()):
        tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def is_affine_in(expr, t):
    """True if expr is a*t + b (degree <= 1) in t."""
    return expr.has(t) and sympy.diff(expr, t, 2) == 0


def heaviside_switchpoints(expr, t):
    """Collect switchpoints t0 where Heaviside(a*t+b) atoms switch (arg==0)."""
    pts = set()
    for h in expr.atoms(sympy.Heaviside):
        arg = sympy.expand(h.args[0])
        if not arg.has(t):
            continue

        if not is_affine_in(arg, t):
            raise ValueError("Heaviside argument {0} is not affine in t".format(arg))
        sol = sympy.solve(sympy.Eq(arg, 0), t)
        if sol:
            pts.add(sympy.simplify(sol[0]))
    return sorted(pts, key=lambda x: float(x))


"""
Marcus kinetics
"""


def marcus_integral(theta, lambda_x):
    """Marcus-Hush-Chidsey integral at nondimensional overpotential `theta`."""
    def integrand(eps):
        return np.exp(
            -np.square(eps - theta) / (4 * lambda_x)
            - np.logaddexp(eps / 2, -eps / 2)
        )

    lo, _ = quad(integrand, -np.inf, theta, limit=200)
    hi, _ = quad(integrand, theta, np.inf, limit=200)
    return lo + hi


def even_coefficient_count(poly_degree):
    """Coefficients in the even fit that replace a `poly_degree` full polynomial.

    An even polynomial with m coefficients spans theta^0 ... theta^(2m-2), so
    matching the highest power of a `poly_degree`-coefficient full polynomial
    (theta^(poly_degree-1)) takes m = ceil(poly_degree/2) rounded up by one.
    """
    return (poly_degree + 2) // 2


def marcus_coefficients(lambda_x, k0, overpotential_range, poly_degree=15):
    """Even-polynomial coefficients of the shared part of both rate constants.

    The MHC integral is even -- substituting eps -> -eps in

        I(theta) = int exp(-(eps-theta)^2 / 4 lambda) / (2 cosh(eps/2)) d eps

    gives I(-theta) == I(theta) -- and the +-theta/2 split is exact, so

        log k_red = log(k0) - theta/2 + g(theta)
        log k_ox  = log(k0) + theta/2 + g(theta),   g(theta) = log(I(theta)/I(0))

    with g even. Fitting g in u = theta^2 therefore needs half the coefficients
    of the two full polynomials it replaces, at identical accuracy, and the one
    fit serves both directions. log(k0) is folded into the constant term, which
    is exact for the same reason: a constant shift lands wholly in order 0.

    Only |theta| is quadratured, since g is even -- the negative half would
    contribute duplicate points to the least squares and nothing else.

    Args:
        lambda_x (float): reorganisation energy (RT/F)
        k0 (float): standard rate constant
        overpotential_range (array): window the polynomial is fitted over
        poly_degree (int): coefficients the equivalent full polynomial would have

    Returns:
        numpy.ndarray: coefficients of g in u = theta^2, lowest order first
    """
    theta = np.unique(np.abs(np.asarray(overpotential_range, dtype=float)))

    I_0 = marcus_integral(0.0, lambda_x)
    g = np.log(np.array([marcus_integral(t, lambda_x) for t in theta]) / I_0)

    # polyfit normalises the Vandermonde columns; a plain lstsq on a polyvander
    # does not, and at these degrees that is the difference between a usable fit
    # and a useless one.
    coeffs = np.polyfit(
        np.square(theta), g, deg=even_coefficient_count(poly_degree) - 1
    )[::-1]
    coeffs[0] += np.log(k0)
    return coeffs


"""
Pre-made input waveforms
"""


def _ramp(name):
    """Triangular sweep, optionally with a superimposed sinusoid (FTACV)."""
    t, Estart, Ereverse, v = sympy.symbols("t Estart Ereverse v")
    thalf = (Ereverse - Estart) / v
    E = 0
    for k in range(0, 2):
        if k % 2 == 0:
            segment = Estart + v * (t - (k * thalf))
        else:
            segment = Ereverse - v * (t - (k * thalf))
        # Heaviside(x, 1) pins H(0)=1 (sympy's default is 1/2, which halves
        # the ramp term at t=0 and puts E(0) at Estart/2 instead of Estart).
        # The gates stay disjoint: at t=k*thalf the previous gate closes as
        # this one opens, so no piece is counted twice.
        gate = sympy.Heaviside(t - (k * thalf), 1) - sympy.Heaviside(
            t - ((k + 1) * thalf), 1
        )
        E += segment * gate
    if name == "FTACV":
        Eamp, phase, omega = sympy.symbols("Eamp phase omega")
        E += Eamp * sympy.sin(omega * t + phase)
    return E


def _staircase(params, label, smoothed):
    """Square-wave staircase, with either Heaviside or tanh-smoothed gates.

    `smoothed=True` replaces each step H(t-a) with 0.5*(1 + tanh((t-a)/eps)),
    giving a square wave with a continuous, finite derivative everywhere. That
    flows through the smooth branch of `render_potential`, so no stop_i / E_i[N]
    indexing is needed and Cdl*dE/dt stays finite.

    The waveform is accumulated as a running sum of transitions rather than as
    one bump per half period. A bump per half period is wrong at both ends: the
    first gate sits exactly on its own transition at t=0, where tanh(0)=0 makes
    it only half open, and the last gate closes again once the scan is over. So
    E started at (Estart+Eamp)/2 rather than Estart+Eamp -- 0.30 V out for a
    -0.6 V start -- and decayed to 0 V after the final step instead of holding
    Estop. Both artefacts also leaked into the Marcus fit window, since
    `build_solver` takes min/max of E over the requested times. Summing the
    jumps instead puts no transition at t=0 at all: E starts at the first level
    and holds the last one.
    """
    t = sympy.symbols("t")
    Estart, E_step, E_amp, omega = sympy.symbols("Estart Estep Eamp omega")
    needed_params = set(["Estart", "Estop", "Estep"])
    missing = needed_params - set(params.keys())
    if missing:
        raise ValueError("For {0} declaration need {1}".format(label, missing))
    N = int(np.ceil(abs(params["Estart"] - params["Estop"]) / (params["Estep"])))
    sign = np.sign(params["Estop"] - params["Estart"])
    tau = 1 / omega*2*np.pi  # full period per step (forward + reverse pulse)
    tp = tau / 2
    eps = sympy.symbols("alpha") * tp if smoothed else None

    # Potential held over each half period, in order: step n sits at base + Esw
    # for [n*tau, n*tau + tp) and base - Esw for [n*tau + tp, (n+1)*tau).
    levels = []
    for n in range(N):
        base = Estart + sign * (n * E_step)  # staircase level for this step
        levels += [base + E_amp, base - E_amp]

    E = levels[0]
    for k in range(1, len(levels)):
        switch = k * tp  # transition from levels[k-1] to levels[k]
        if smoothed:
            step = (1 + sympy.tanh((t - switch) / eps)) / 2
        else:
            # H(0)=1 (see _ramp above) so a transition takes effect on its own
            # sample rather than the one after it
            step = sympy.Heaviside(t - switch, 1)
        E += (levels[k] - levels[k - 1]) * step
    return E


def waveform(name, params=None):
    """Build a pre-made input potential expression.

    Args:
        name (str): one of DCV, FTACV, SWV, SWVtanh
        params (dict, optional): required by the staircase forms, which need
            Estart/Estop/Estep at build time to know how many steps to emit

    Returns:
        sympy expression in `t` and the waveform's symbolic parameters
    """
    params = {} if params is None else params
    if name in ("FTACV", "DCV"):
        return _ramp(name)
    if name == "SWV":
        return _staircase(params, "SWV", smoothed=False)
    if name == "SWVtanh":
        return _staircase(params, "SWVtanh", smoothed=True)
    raise ValueError(
        "Pre defined input can be one of DCV, FTACV, SWV, SWVtanh, not {0}".format(name)
    )
