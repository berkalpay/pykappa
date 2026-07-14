# %% [markdown]
# # Tutorial
#
# PyKappa works in terms of patterns, mixtures, and rules, and expressions, culminating in systems.
# Let's see how the API works at each level, including internal implementation details.

# %% nbsphinx="hidden"
import random

random.seed(0)

# %% [markdown]
# ## Patterns
#
# As in Kappa, patterns (subsets of a mixture on which rules can operate) can be broken down into components (a connected set of agents).
# Agents are typed objects with named sites, each of which has an internal state and possibly a binding partner -- the site of another agent.
# All of these classes of objects can be constructed from Kappa strings using the `from_kappa` classmethod.

# %%
from pykappa import Agent, Pattern, Component

# Parse agents A and B from Kappa strings
a = Agent.from_kappa("A(x[.])")
b = Agent.from_kappa("B(x[.])")
print("Agent A:", a.kappa_str)
print("Agent B:", b.kappa_str)

# Create an AB complex
complex = Component.from_kappa("A(x[1]), B(x[1])")
# Or equivalently:
complex = Pattern.from_kappa("A(x[1]), B(x[1])").components[0]
print("Complex", complex.kappa_str)

# %% [markdown]
# Each agent in a mixture is essentially a node in a graph.
# PyKappa implements functions such as the ones below, identifying neighbors of an agent and checking whether a component embeds in another, or is in other words isomorphic to a subset of another.

# %%
root = next(iter(complex))
print("Root agent type:", root.type)
print("Neighbors of root:", [neighbor.type for neighbor in root.neighbors])

self_embeddings = list(complex.embeddings(complex))
print("Embeddings of complex in itself:", len(self_embeddings))

# %% [markdown]
# ## Mixtures
#
# A mixture, like components and patterns, is a collection of agents, but one which facilitates the application of rules by efficiently updating embeddings according to changes in the mixture.
# Mixtures can be initialized and adjusted programatically, or again initialized from Kappa strings.

# %%
from pykappa import Mixture

mixture = Mixture()
mixture.add("A(x[.])", n_copies=3)
mixture.add("B(x[.])", n_copies=2)
mixture.add("A(x[1]), B(x[1])", n_copies=2)

print(f"Mixture as a Kappa string:\n{mixture.kappa_str}\n")

# Track a component pattern to query embeddings efficiently
mixture._track_component(complex)
print("#AB embeddings (cached):", len(mixture.embeddings(complex)))

# %% [markdown]
# By default, the mixture is unaware of the components of which it is composed: it has an agent-level view that is sufficient for simulation of basic rule applications.
# Some types of rules, such as those with distinct unimolecular and bimolecular rates, require component-level information, in which case component-tracking will be automatically enabled at system initialization.
# Enabling component-tracking also allows for efficient component-level queries.

# %%
comp_mixture = Mixture(track_components=True)
comp_mixture.add("A(x[.])", n_copies=2)
comp_mixture.add("A(x[1]), B(x[1])")

# Iterate over all components
print("Components in mixture:")
for component in comp_mixture:
    print(component.kappa_str)

# Query embeddings within a specific component
comp_mixture._track_component(complex)
mixture_component = next(c for c in comp_mixture.components if len(c.agents) == 2)
embeddings_in_comp = comp_mixture.embeddings_in_component(complex, mixture_component)
print(f"\nEmbeddings in the AB complex: {len(embeddings_in_comp)}")

# %% [markdown]
# ## Rules
#
# Rules transform agents matched by their left-hand side into those specified by the right-hand side.
# Specifically, `rate(system)` evaluates the stochastic rate (possibly using variables), `n_embeddings(mixture)` counts applicable embeddings, and `select(mixture)` samples a specific embedding and returns a MixtureUpdate, which a mixture can then take to efficiently apply the corresponding update.

# %% [markdown]
# Let's add a rule to bind `A` and `B` into `AB` and see how the mixture applies it.

# %%
from pykappa import Rule

print("AB count before one binding event:", len(mixture.embeddings(complex)))

bind = Rule.from_kappa("A(x[.]), B(x[.]) -> A(x[1]), B(x[1]) @ 1")

# Track the components on the left-hand side of the rule
for component in bind.left.components:
    mixture._track_component(component)

update = bind._select(mixture)
mixture._apply_update(update)

mixture._track_component(complex)
print("AB count after one binding event:", len(mixture.embeddings(complex)))

# %% [markdown]
# ## Expressions
#
# Expressions in PyKappa represent algebraic formulas that can include literals, variables, operators, and component patterns.
# They are used for rule rates, observables, and variables.
# Expressions can be parsed from Kappa strings and evaluated in the context of a system.

# %%
from pykappa._expression import Expression

literal_expr = Expression.from_kappa("42")
print("Literal expression value:", literal_expr.evaluate())

math_expr = Expression.from_kappa("(2 + 3) * 4")
print("Math expression value:", math_expr.evaluate())

pattern_expr = Expression.from_kappa("|A(x[1]), B(x[1])|")
print("Pattern expression:", pattern_expr.kappa_str)

# %% [markdown]
# The last expression represents the number of AB complexes; it can be evaluated given a system as `pattern_expr.evaluate(system)`.

# %% [markdown]
# ## Systems
#
# A system bundles a mixture with rules and observables and is used for simulation.
# Start with the reversible binding system in the Examples gallery to see how the API works at this highest level.
