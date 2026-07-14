# %% [markdown]
# # Reversible binding

# %% [markdown]
# We can initialize a system of a simple reversible binding interaction as follows:

# %%
from pykappa import System

system = System.from_kappa(
    mixture={"A(x[.])": 100, "B(x[.])": 100},
    rules=[
        "A(x[.]), B(x[.]) -> A(x[1]), B(x[1]) @ 1",
        "A(x[1]), B(x[1]) -> A(x[.]), B(x[.]) @ 1",
    ],
    observables={"AB": "|A(x[1]), B(x[1])|"},
    seed=42,
)

# %% [markdown]
# or equivalently from a .ka-style string:

# %%
system = System.from_ka(
    """
    %init: 100 A(x[.])
    %init: 100 B(x[.])

    %obs: 'A' |A(x[.])|
    %obs: 'AB' |A(x[1]), B(x[1])|

    A(x[.]), B(x[.]) <-> A(x[1]), B(x[1]) @ 1, 1
    """,
    seed=42,
)

# %% [markdown]
# 100 instances of molecules of type `A` and of type `B`, each with an empty binding domain `x`, are created, and we track the number of `AB` complexes.
#
# We're going to simulate this system and plot its behavior, marking certain times of interest. We'll first simulate until time 1:

# %%
times = []
while system.time < 1:
    system.update()
times.append(system.time)

# %% [markdown]
# We'll now manually add 50 new `A` and `B` molecules each and simulate until there are no more than 10 free `A` in the mixture:

# %%
system.mixture.add("A(x[.]), B(x[.])", 50)

while system["A"] > 10:
    system.update()
times.append(system.time)

# %% [markdown]
# Now let's simulate some more time:

# %%
while system.time < 2:
    system.update()
times.append(2)

# %% [markdown]
# The default simulator provides the most features since it’s written directly in Python, but models can be offloaded to [KaSim](https://github.com/Kappa-Dev/KappaTools), a compiled Kappa simulator, for faster simulation. For example, we could've run:
# ```python
# system.update_via_kasim(time=1)
# ```

# %% [markdown]
# Finally, let’s plot the history of the quantities we tracked:

# %%
import matplotlib.pyplot as plt

system.monitor.plot(combined=True)
for time in times:
    plt.axvline(time, color="black", linestyle="dotted")
plt.show()

# %% [markdown]
# The system equilibrates relatively early.
# Then new `A` is added and the number of `AB` complexes increases, reaching a higher equilibrium concentration.
