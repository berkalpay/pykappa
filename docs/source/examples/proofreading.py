# %%

# %% [markdown]
# # Kinetic Proofreading
#
# This example implements the concept of *kinetic proofreading* [first
# formalized](https://doi.org/10.1073/pnas.71.10.4135) by John Hopfield 
# in 1974. This notebook is necessarily longer and more complicated than
# the others: it implements a complex biochemical model. If you are new
# to PyKappa, familiarize yourself with our
# [tutorial](https://pykappa.org/tutorial.html) and early
# [examples](https://pykappa.org/examples/reversible_binding.html). This
# example also makes use of advanced Python language features. Further
# reading material is supplied at the end of this notebook.
#
# ## Why Kinetic Proofreading?
#
# Biological systems have a vested interest in preserving genetic
# material across transcription, translation, and duplication. Small
# errors at any point can have drastic impacts on downstream function.
# At thermodynamic equilibrium, the error rate will be based only on the
# difference in free energy between correct and incorrect pairings: it
# will go as $\exp(-(E_W - E_R) / k_B T)$, where $E_W$ is the binding
# energy of an incorrect monomer pair (e.g. attaching a guanine to a
# cytosine) and $E_R$ is the binding energy of a correct pair.
#
# Unfortunately, differences between structures used in copying are
# small enough as to render pure energy discrimination impossible: the
# error fraction will be unacceptably high. Indeed, copying reactions
# must utilize potentially several out-of-equilibrium intermediate steps
# to aggressively reduce the error rate.
#
# In this example, following Pigolotti and Sartori's "[Protocols for
# Copying and Proofreading](https://doi.org/10.1007/s10955-015-1399-2)"
# &#167;6, we use just two: an initial monomer addition step after which the copy
# machine must "commit" the change. This means that each monomer
# addition has to undergo *two* reactions, which has the effect of
# multiplicatively enhancing the error rate.
#
# <table><tr>
# <td><img src="https://github.com/user-attachments/assets/fcaa132e-cf4d-4825-b440-640953d72c1f" alt="diagram"></td>
# <td><img src="https://github.com/user-attachments/assets/fcaa132e-cf4d-4825-b440-640953d72c1f" alt="diagram"></td>
# </tr></table>

# %%
import copy
import numpy as np
import matplotlib.pyplot as plt

from string import Template
from pykappa import System, Mixture

rules = []
rng = np.random.default_rng(seed=42)

# THERMODYNAMIC CONSTANTS
T = 25.0
Avogadro = 6.0221413e23
R = 0.008314
T0 = 273.15
RT = (T0 + T) * R

# %% [markdown]
# !EXPLAIN CONTACT MAP
#
# The first rule we should specify is how the copy machine should behave
# at the beginning of the polymer. It will be bound at initialization to
# the template strand. But one problem presents itself. The rates at
# which a copy machine should attach a new monomer should depend on
# whether or not that monomer's type matches that of the template. Thus,
# we need four different rules for each of the possible combinations. (A
# system with four possible bases requires sixteen rule enumerations.)
#
# Thankfully, Python makes it easy to generate large rulesets. Kappa
# rules are particularly compatible with Python's
# [\$-strings](https://docs.python.org/3/library/string.html#template-strings-strings),
# which consume `$` characters and thus do not conflict with Kappa's
# existing syntax. A simple function generates four rules for each
# pairing: `a` is the first base, `b` is the second, and `c` is
# `"correct"` if the bases match and `"incorrect"` otherwise. We use the
# following function to generate rulesets:


# %%
def generate_ruleset(rule):
    tmpl = Template(rule)
    for pair in ["ww", "bb", "wb", "bw"]:
        c = "correct" if pair[0] == pair[1] else "incorrect"
        yield tmpl.safe_substitute(a=pair[0], b=pair[1], c=c).strip()


# %% [markdown]
# Now we are ready to write the rule itself. In words:
#
# *Begin the construction of a transcript by attaching a monomer to the
# copy machine's blank transcript site. This addition is tentative and
# may be reversed; increment the state variable.*

# %%
rules += generate_ruleset(
    "M(l[.], t{$a}[2], r[1]), M(l[1], t[.]), "
    "., C(c{0}[.], t[2]) "
    "<-> "
    "M(l[.], t{$a}[.], r[1]), M(l[1], t[2]), "
    "M(l[.], t{$b}[.], r[3]), C(c{1}[3], t[2]) "
    "@ 'add_${c}_f', 'add_${c}_r'"
)

# %% [markdown]
# In general, the first line of the rule specifies the *template*
# string; the second line specifies the *transcript* string including
# the copy machine.
#
# The result of `generate_ruleset` will be a set of four rules (actually
# eight if we count the reverse cases) where `$a` and `$b` have been
# substituted with the value of the template base and the transcript
# base, respectively. These rules will fire at different rates depending
# on whether the base pairs match. This boolean is given as `$c`.
#
# We also utilize Python's [string literal
# concatentation](https://docs.python.org/3/reference/expressions.html#string-concatenation)
# to increase the readability of the rule by splitting it over multiple
# lines. Indeed, these strings are equivalent:

# %%
"A" "B" == "AB"

# %% [markdown]
# But this choice is only for the sake of readability: rule
# functionality is not impacted. In fact, PyKappa will happily parse
# rules which span multiple lines.
#
# We need rules to handle the similar case whereby a new monomer is
# attached to an existing transcript strand. That is, we need to handle
# the case where the copy machine is in the *middle* of the chain. In
# words:
#
# *Given a committed copy machine, append a new monomer to an existing
# transcript strand. Increment the state variable to allow for
# reversal.*

# %%
rules += generate_ruleset(
    "M(t{$a}[2], r[1]), M(l[1], t[.]), "
    "M(r[3]), ., C(c{0}[3], t[2]) "
    "<-> "
    "M(t{$a}[.], r[1]), M(l[1], t[2]), "
    "M(r[4]), M(l[4], t{$b} r[3]), C(c{1}[3], t[2]) "
    "@ 'add_${c}_f', 'add_${c}_r'"
)

# %% [markdown]
# Once the copy machine reaches the end of the strand, we should
# provide some instructions for how to behave. Simply:
#
# *When the copy machine reaches the end of the template, i.e. it is
# bound to a capping monomer, attach a cap to the transcript and fall
# off.*

# %%
rules.append(
    "M(t{cap}[1]), "
    "M(r[2]), ., C(c{0}[2], t[1]) "
    "-> "
    "M(t{cap}[.]), "
    "M(r[1]), M(l[1], t{cap}), C(c{0}[.], t[.]) "
    "@ 'add_correct_f'"
)

# %% [markdown]
# Because each of these rule classes concern monomer addition (occuring
# at the beginning, middle, or end), they share constants. Instructions
# for reaching the end of the strand is not a ruleset; the rate should
# be identical no matter the type of the penultimate monomer.
#
# The machine will have to "commit" any monomer addition it makes by
# returning the state of its `c` site to zero. This checkpoint amplifies
# the discriminative capability of the copy machine. This rule must fire
# before the machine is allowed to continue forward. (There is, of
# course, the exception of the proofreading rules, which we will cover
# later.)
#
# *Commit a transcript change by chaning the copy machine's state to
# zero.*

# %%
rules += generate_ruleset(
    "M(t{$a}, r[1]), M(l[1], t[2]), "
    "M(t{$b}, r[3]), C(c{1}[3], t[2]) "
    "<-> "
    "M(t{$a}, r[1]), M(l[1], t[2]), "
    "M(t{$b}, r[3]), C(c{0}[3], t[2]) "
    "@ 'commit_${c}_f', 'commit_${c}_r'"
)

# %% [markdown]
# The ruleset is complete; all that remains is to specify the rates
# themselves. In general, one should prefer to *generate* rates by
# specfiying free energies. This ensures that these systems do not
# violate any rules of thermodynamics.
#
# !INSERT LONGER FREE ENERGY EXPLANATION

# %%

# FREE ENERGIES
R0 = 0.0
W0 = 0.0
R1 = 0.0
W1 = 1.0
R2 = 0.0
W2 = 1.0


def get_rates(r0, w0, r1, w1, barrier=0.0, drive=1.0, omega=1.0):
    return {
        "correct_f": omega * np.exp((r0 + drive + barrier) / RT),
        "incorrect_f": omega * np.exp((w0 + drive) / RT),
        "correct_r": omega * np.exp((r1 + barrier) / RT),
        "incorrect_r": omega * np.exp(w1 / RT),
    }


def vars_of_rates(t0, t1, pr):
    vmap = {"add": t0, "commit": t1, "proofread": pr}
    return {
        f"{name}_{key}": rate
        for name, rates in vmap.items()
        for key, rate in rates.items()
    }


# %% [markdown]
# Now that we have a collection of rules, we'll need to write a function
# to generate a polymer for the system to use. We'll perform our
# proofreading experiments with the same polymer copied for many new
# systems.


# %%
def generate_polymer(length):
    assert length > 1
    pattern = "".join(rng.choice(["w", "b"], length))

    agents = ["C(c{0}[.], t[1])", f"M(l[.] r[2] t{{{pattern[0]}}}[1])"]
    for i, base in enumerate(pattern[1:]):
        agents.append(f"M(l[{i+2}] r[{i+3}] t{{{base}}}[.])")
    agents.append(f"M(l[{length+1}] r[.] t{{cap}}[.])")

    return ", ".join(agents), pattern


def walk_polymer(component):
    head = lambda m: m.type == "M" and not m["l"].bound
    monomer = next((m for m in component if head(m)), None)
    if not monomer:
        return None

    transcript = ""
    while monomer["t"].state != "cap":
        transcript += monomer["t"].state
        monomer = monomer["r"].partner.agent

    return transcript


def get_transcript(mixture, pattern):
    transcript = None
    for component in mixture:
        if (p := walk_polymer(component)) and p != pattern:
            transcript = p
    return transcript or pattern

# %% [markdown]
# !EXPLAIN
#
# t0 = get_rates(R0, W0, R1, W1, barrier=1.0, drive=1.0)
# t1 = get_rates(R0, W0, R1, W1, barrier=1.0, drive=1.0)
# pr = get_rates(R0, W0, R2, W2, omega=0.0)
#
# polymer, pattern = generate_polymer(25)
# variables = {k: str(v) for k, v in vars_of_rates(t0, t1, pr).items()}
#
# system = System.from_kappa(
#     rules=rules, variables=variables, mixture={polymer: 1}, seed=42
# )
#
# while system.reactivity:
#     system.update()
#
# transcript = get_transcript(system.mixture, pattern)
#
# print("base:      ", pattern)
# print("transcript:", transcript)
# print("errors:    ", sum(x != y for x, y in zip(pattern, transcript)))
# print("time:      ", system.time)

# %% [markdown]
# Now that we can count errors, we can examine how changing the
# configuration of the free energy landscape affects the effectiveness
# of the copy machine.


# %%
class Context:
    def __init__(self, base, polymer, pattern):
        self.base = base
        self.polymer = polymer
        self.pattern = pattern
    
    def new(self):
        mixture = Mixture([copy.copy(self.polymer)])
        return System(rules=self.base.rules.values(), 
                      variables=self.base.variables, 
                      mixture=mixture)

    def count_errors(self, mixture):
        transcript = get_transcript(mixture, self.pattern)
        return sum(x != y for x, y in zip(pattern, transcript))


def experiment(context, b0, b1):
    t0 = get_rates(R0, W0, R1, W1, barrier=b0, drive=1.0)
    t1 = get_rates(R1, W1, R2, W2, barrier=b1, drive=1.0)
    pr = get_rates(R0, W0, R2, W2, omega=0.0)

    system = context.new()
    for k, v in vars_of_rates(t0, t1, pr).items():
        system[k] = v

    while system.reactivity:
        system.update()

    return context.count_errors(system.mixture)

# %% [markdown]
# !EXPLAIN

# %%
# polymer, pattern = generate_polymer(10)
# system = System.from_kappa(rules=rules, variables=variables, mixture={polymer: 1})
# context = Context(system, polymer, pattern)
# 
# gridsize = 5
# trials = 10
# barrier_grid = np.linspace(0.0, 10.0, gridsize)
# 
# B0, B1 = np.meshgrid(barrier_grid, barrier_grid)
# B0 = np.repeat(B0[..., None], trials, axis=-1)
# B1 = np.repeat(B1[..., None], trials, axis=-1)
# 
# parallel = Parallel(n_jobs=-1, verbose=10)
# out = parallel(
#     delayed(experiment)(context, B0[idx], B1[idx])
#     for idx in np.ndindex(B0.shape)
# )

# results = np.mean(np.array(out).reshape(B0.shape), axis=-1)

# %% [markdown]
# !EXPLAIN

# %%
# fig, ax = plt.subplots(figsize=(7, 6))
# im = ax.imshow(
#     results,
#     origin="lower",
#     aspect="equal",
#     cmap="inferno_r",
#     extent=[barrier_grid[0], barrier_grid[-1],
#         barrier_grid[0], barrier_grid[-1]],
# )
# ax.set_xlabel("t0 barrier (0 \u2194 1)")
# ax.set_ylabel("t1 barrier (1 \u2194 2)")
# ax.set_title("Mean error rate vs. transition barriers (proofreading off)")
# fig.colorbar(im, label="mean error rate")
# plt.tight_layout()
# plt.show()
