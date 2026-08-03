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
# But differences between structures used in copying are small enough as to
# render pure energy discrimination impossible: the error fraction will be
# unacceptably high. Indeed, copying reactions must utilize potentially several
# out-of-equilibrium intermediate steps to aggressively reduce the error rate.
#
# In this example, following Pigolotti and Sartori's "[Protocols for
# Copying and Proofreading](https://doi.org/10.1007/s10955-015-1399-2)"
# &#167;6, we use just two: an initial monomer addition step after which the copy
# machine must "commit" the change. This means that each monomer
# addition has to undergo *two* reactions, which has the effect of
# multiplicatively enhancing the error rate.

# %% [raw] raw_mimetype="text/restructuredtext"
# .. image:: https://github.com/user-attachments/assets/fcaa132e-cf4d-4825-b440-640953d72c1f
#    :alt: diagram

# %% [raw] raw_mimetype="text/restructuredtext"
# .. image:: https://github.com/user-attachments/assets/1d8702fd-8a96-47af-a6de-281d9fb1d4c1
#    :alt: free energy landscape
#    :align: center
#    :width: 360px

# %% [markdown]
# We sketch the three reactions above and the corresponding free energy
# landscape below. The dotted line represents the backwards reaction. We
# begin by importing relevant packages and defining global variables
# which will become important later.

# %%
import copy
import numpy as np
import matplotlib.pyplot as plt

from string import Template
from pykappa import System, Mixture

rules = []
rng = np.random.default_rng(seed=42)

# %% [markdown]
# Below we explicitly define the energy landscape (`R1` corresponds to
# $E_1^R$, `W2` to $E_2^W$, etc.) so that our rate constants do not
# violate the Second Law of Thermodynamics.

# %%
# free energies
R0 = 0.0
W0 = 0.0
R1 = 0.0
W1 = 1.0
R2 = 0.0
W2 = 1.0

# thermodynamic constants
T = 25.0
Avogadro = 6.0221413e23
R = 0.008314
T0 = 273.15
RT = (T0 + T) * R

# %% [markdown]
# The function below returns a set of four rate constants for one of the
# three reactions listed above. The derivation for these formulae is
# given at the end of this document. The rates at which a copy machine
# should attach a new monomer should depend on whether or not that
# monomer's type matches that of the template.


# %%
def get_rates(r0, w0, r1, w1, barrier=0.0, drive=1.0, omega=1.0):
    return {
        "correct_f": omega * np.exp((r0 + drive + barrier) / RT),
        "incorrect_f": omega * np.exp((w0 + drive) / RT),
        "correct_r": omega * np.exp((r1 + barrier) / RT),
        "incorrect_r": omega * np.exp(w1 / RT),
    }


# %% [markdown]
# ## The Model
#
# Now that we have a general understanding of the model's reactions and
# function, we set out to define more explictily the function of this
# model. The contact map for this model will be:

# %% [raw] raw_mimetype="text/restructuredtext"
# .. image:: https://github.com/user-attachments/assets/00e6171e-39f1-40d5-b069-e8bf26e9c616
#    :alt: contact map
#    :align: center
#    :width: 320px

# %% [markdown]
# We specify a copy machine `C` as an abstraction for somthing like RNA
# polymerase. We also introduce a monomer, `C`, which carries a state
# specifying its encoding. (In the image, we use a closed circle and an
# open circle; in code, we will use `w` and `b`.)
#
# The copy machine interacts with `M` in two ways: by binding the
# template string (the reference polymer) via site `t` or the transcript
# (the new polymer) via site `c`. The copy machine's job is to build a
# growing string by matching base pairs, starting at the beginning.
#
# The first rule we should specify is how the copy machine should behave
# at the beginning of the polymer. It will be bound at initialization to
# the template strand. We will use four different rules for each
# possible template-transcript base pairing. (A system with four
# possible bases requires sixteen rule enumerations.)
#
# Thankfully, Python makes it easy to generate large rulesets. Kappa
# rules are particularly compatible with Python's
# [\$-strings](https://docs.python.org/3/library/string.html#template-strings-strings),
# which consume `$` characters and thus do not conflict with Kappa's
# existing syntax. The simple function below generates four rules for each
# pairing: `a` is the first base, `b` is the second, and `c` is
# `"correct"` if the bases match and `"incorrect"` otherwise.


# %%
def generate_ruleset(rule):
    tmpl = Template(rule)
    for pair in ["ww", "bb", "wb", "bw"]:
        c = "correct" if pair[0] == pair[1] else "incorrect"
        yield tmpl.safe_substitute(a=pair[0], b=pair[1], c=c).strip()


# %% [markdown]
# For example, this simple "rule" (this is not valid Kappa) can be
# expanded into four:

# %%
print(*generate_ruleset("$a, $b @ $c"), sep="\n")

# %% [markdown]
# The pattern should be clear. Now we are ready to write the rules
# themselves. In words:
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
# As they say, a picture is worth a thousand words. Here's a
# diagram which exactly corresponds to the text of the rule, applied
# specifically to a b-w pairing:

# %% [raw] raw_mimetype="text/restructuredtext"
# .. image:: https://github.com/user-attachments/assets/12de2888-149c-4a65-bb3d-0953e9889332
#    :alt: rule

# %% [markdown]
# Note that we exclude some sites and states here. In particular, we
# leave the state on the top left monomer's `t` site ambiguous; the same
# monomer has its `r` site unspecified entirely. This means that `t` may
# have *any* state. This is also true for `r`, but by leaving it out of the
# pattern entirely, we also allow it to be bound or unbound. (We take
# care to specify `t`'s binding.) This is the *don't care, don't write*
# principle of Kappa, and is designed to make rules a little easier to
# read and write.
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
# Now we will define the proofreading reaction. It should be capable of
# removing a committed monomer from the transcript chain.
#
# *Undo the addition of a commited monomer.*

# %%
rules += generate_ruleset(
    "M(t{$a}[.], r[1]), M(l[1], t[2]), "
    "M(r[4]), M(l[4], t{$b} r[3]), C(c{0}[3], t[2]) "
    "<-> "
    "M(t{$a}[2], r[1]), M(l[1], t[.]), "
    "M(r[3]), ., C(c{0}[3], t[2]) "
    "@ 'proofread_${c}_f', 'proofread_${c}_r'"
)

# %% [markdown]
# Finally,
#
# *Remove the first committed monomer via proofreading.*

# %%
rules += generate_ruleset(
    "M(l[.], t{$a}[.], r[1]), M(l[1], t[2]), "
    "M(l[.], t{$b}[.], r[3]), C(c{0}[3], t[2]) "
    "<-> "
    "M(l[.], t{$a}[2], r[1]), M(l[1], t[.]), "
    "., C(c{0}[.], t[2]) "
    "@ 'proofread_${c}_f', 'proofread_${c}_r'"
)

# %% [markdown]
# The ruleset is complete; all that remains is to specify the rates
# themselves. We do this by introducing a new helper function
# `vars_of_rates`, which converts three sets of rate constants to the
# Kappa variables referenced by our rules.


# %%
def vars_of_rates(t0, t1, pr):
    vmap = {"add": t0, "commit": t1, "proofread": pr}
    return {
        f"{name}_{key}": rate
        for name, rates in vmap.items()
        for key, rate in rates.items()
    }


# %% [markdown]
# ## Simulation
#
# Now that we have a collection of rules, we'll need to write a function
# to generate a polymer for the system to use, along with functions
# which identify the number of errors in the new transcript strand.


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
# A complete experiment follows. Note that the barrier on `pr` is
# negative: we want the reaction to be *more* likely for incorrect
# monomer pairs.

# %%
t0 = get_rates(R0, W0, R1, W1, barrier=1.0, drive=1.0)
t1 = get_rates(R1, W1, R2, W2, barrier=1.0, drive=1.0)
pr = get_rates(R2, W2, R0, W0, barrier=-1.0, drive=1.0)

polymer, pattern = generate_polymer(25)
variables = {k: str(v) for k, v in vars_of_rates(t0, t1, pr).items()}

system = System.from_kappa(
    rules=rules, variables=variables, mixture={polymer: 1}, seed=42
)

while system.reactivity:
    system.update()

transcript = get_transcript(system.mixture, pattern)

print("base:      ", pattern)
print("transcript:", transcript)
print("errors:    ", sum(x != y for x, y in zip(pattern, transcript)))
print("time:      ", system.time)

# %% [markdown]
# One might extend this example by systematically varying different rate
# parameters and measuring how system behavior changes in response. For
# example, one might expect that making reaction barriers more extreme
# would result in a lower error rate without a time penalty.
#
# ## Reading Material
# - Pigolotti, S. & Sartori, P. Protocols for Copying and Proofreading
# in Template-Assisted Polymerization. J Stat Phys 162, 1167–1182
# (2016). ([URL](https://doi.org/10.1007/s10955-015-1399-2))
#
# - Hopfield, J. J. Kinetic Proofreading: A New Mechanism for Reducing
# Errors in Biosynthetic Processes Requiring High Specificity. Proc Natl
# Acad Sci U S A 71, 4135–4139 (1974).
# ([URL](https://pmc.ncbi.nlm.nih.gov/articles/PMC434344/))
