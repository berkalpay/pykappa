# ---
# jupyter:
#   language_info:
#     name: python
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   nbsphinx:
#     execute: never
# ---

# %% [markdown]
# # Phase separation
#
# Multivalent binding allows molecules to assemble into large connected complexes.
# Here, `A` can polymerize through its `l` and `r` sites and bind `B` through its
# `b` site. Each `B` has three `A`-binding sites and a dimerization site `d`.
# We will simulate the system and track the largest connected complex, the *maximer*.

# %%
import base64
import zlib
from pathlib import Path

import pandas as pd

from pykappa import System

# %% [markdown]
# Convert concentrations to molecule counts and kinetic rate constants to
# stochastic rates, and set up the model.

# %%
scale = 0.03 * 0.25
volume = 1e-12 * scale
avogadro = 6.022e23
alpha = 0.67e5 * scale
k_on = 1e8
initial_count = int(100 * (1e-9 * avogadro * volume))
g_on = k_on / (avogadro * volume)
g_on_local = alpha * g_on
g_off_medium = 1e-7 * k_on
g_off_weak = 1e-6 * k_on

model = f"""
%init: {initial_count} A(l[.], r[.], b[.])
%init: {initial_count} B(d[.], a1[.], a2[.], a3[.])

A(l[.]), A(r[.]) <-> A(l[1]), A(r[1]) @ {g_on} {{{g_on_local}}}, {g_off_weak}
A(b[.]), B(a1[.]) <-> A(b[1]), B(a1[1]) @ {g_on} {{{g_on_local}}}, {g_off_medium}
A(b[.]), B(a2[.]) <-> A(b[1]), B(a2[1]) @ {g_on} {{{g_on_local}}}, {g_off_medium}
A(b[.]), B(a3[.]) <-> A(b[1]), B(a3[1]) @ {g_on} {{{g_on_local}}}, {g_off_medium}
B(d[.]), B(d[.]) <-> B(d[1]), B(d[1]) @ {g_on} {{{g_on_local}}}, {g_off_medium}
"""

system = System.from_ka(model, seed=78746)

# %% [markdown]
# Follow the largest assembly over time, saving system snapshots and its history for later analysis.

# %%
output_directory = Path("phase_separation_output")
output_directory.mkdir(parents=True, exist_ok=True)
observations = []
snapshot_index = 0
end_time = system.time + 33.0
observation_time = snapshot_time = system.time

while system.time < end_time:
    next_update_time = system.next_update_time
    if next_update_time is None:
        break

    if next_update_time >= observation_time:
        maximer = max(system.mixture.components, key=len)
        encoded = base64.b64encode(
            zlib.compress(maximer.kappa_str_with_agent_ids.encode(), level=6)
        ).decode()
        observations.append((round(observation_time, 3), len(maximer), encoded))
        observation_time += 0.005

    if next_update_time >= snapshot_time:
        system.advance_time_to(snapshot_time)
        system.save(output_directory / f"snap_{snapshot_index:06}.pkl")
        snapshot_index += 1
        snapshot_time += 0.01

    system.update()

data = pd.DataFrame(observations, columns=["time", "maximer size", "maximer_strings"])
data.to_pickle(output_directory / "maximer_series.pkl")
