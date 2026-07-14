# %% [markdown]
# # The lac operon
#
# The *E. coli* lac operon controls the production of enzymes needed to metabolize lactose and is regulated in large part by lactose availability.
# This example model includes
# - LacI (`I`) repression of the lac operon via blocking of the operator (`O`),
# - lactose (`L`) transport by LacY (`Y`) permease,
# - conversion of lactose to allolactose (`A`) by LacZ (`Z`),
# - and allolactose-mediated derepression of the operon.

# %%
from pykappa import System
import matplotlib.pyplot as plt

system = System.from_ka(
    """
    %init: 1 O(i[.])
    %init: 1 I(o[.])
    %init: 10 Y()
    %init: 10 Z()

    %obs: 'Extracellular lactose' |L(loc{out})| / 100
    %obs: 'LacY' |Y()|
    %obs: 'Free lac operon' |O(i[.])|

    I(o[.]), O(i[.]) <-> I(o[1]), O(i[1]) @ 1, 0.1   // lac repression
    Y(), L(loc{out})  -> Y(), L(loc{in}) @ 0.01      // lactose transport
    Z(), L(loc{in})   -> Z(), A(z[.], i[.]) @ 1      // lactose to allolactose
    A(i[.]), I(o[.]) <-> A(i[1]), I(o[1]) @ 1, 0.1   // lac derepression
    O(i[.]), ., .     -> O(i[.]), Z(), Y() @ 1       // lac expression
    L(loc{in})        -> . @ 1                       // lactose metabolism

    // degradation
    A() -> . @ 0.1
    Y() -> . @ 0.5
    Z() -> . @ 0.5
    """,
    seed=42,
)

# %% [markdown]
# The rule graph illustrates the local transformations that the rules specify.
# The conditions for these transformations are not detailed in the graph as they are in the Kappa string representation.

# %%
system.rule_graph()

# %% [markdown]
# We simulate the system first without lactose, then add extracellular lactose to observe the regulatory response.

# %%
# Simulate with no lactose
while system.time < 300:
    system.update()

# Add extracellular lactose and continue simulating
system.mixture.add("L(loc{out})", 1000)
while system.time < 800:
    system.update()

# %%
system.monitor.plot(figsize=(4.5, 4.5))
plt.show()

# %% [markdown]
# Without lactose, the operon remains mostly repressed.
# When extracellular lactose is added,
# - existing LacY transports some lactose into the cell,
# - intracellular lactose is converted to allolactose by LacZ,
# - allolactose inactivates the LacI repressor,
# - derepressing the operon and leading to increased LacY and LacZ production.
#
# As lactose is metabolized the operon is again repressed.
