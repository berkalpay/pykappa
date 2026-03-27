import math
import colorsys
import shutil
import tempfile
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from graphviz import Source
import matplotlib.pyplot as plt
import matplotlib.figure

if TYPE_CHECKING:
    from pykappa.pattern import Component
    from pykappa.system import System


class _ComponentPlot:
    """Stable visualization of a Component across simulation steps."""

    def __init__(self, component: "Component"):
        self.component = component
        self._positions: dict[int, tuple[float, float]] = {}

    def _compute_positions(self) -> dict[int, tuple[float, float]]:
        """Assign each agent a fixed position based on identity, computed once."""
        new_agents = [a for a in self.component.agents if id(a) not in self._positions]

        if new_agents:
            # Place new agents on a sunflower spiral (evenly distributed, deterministic)
            n = len(self._positions)
            golden_angle = math.pi * (3 - math.sqrt(5))
            for i, agent in enumerate(new_agents):
                k = n + i
                r = math.sqrt(k + 1)
                angle = k * golden_angle
                self._positions[id(agent)] = (r * math.cos(angle), r * math.sin(angle))

        return {id(a): self._positions[id(a)] for a in self.component.agents}

    def __call__(self, legend: bool = True):
        agent_types = sorted(dict.fromkeys(a.type for a in self.component.agents))
        type_color = {
            t: "#{:02x}{:02x}{:02x}".format(
                *[
                    int(c * 255)
                    for c in colorsys.hls_to_rgb(i / len(agent_types), 0.4, 0.8)
                ]
            )
            for i, t in enumerate(agent_types)
        }

        edges = set()
        for a in self.component.agents:
            for b in a.neighbors:
                if a is b:
                    continue
                edges.add(tuple(sorted((id(a), id(b)))))

        pos = self._compute_positions()

        lines = [
            "graph {",
            "  graph [overlap=false];",
            '  node  [shape=circle, width=0.05, height=0.05, fixedsize=true, label="", style=filled];',
            "  edge  [penwidth=0.3];",
        ]
        if legend:
            min_y = min(y for x, y in pos.values())
            max_x = max(x for x, y in pos.values())
            lx = max_x + 2.0
            legend_vertical_spacing = 0.5
            for i, (t, color) in enumerate(reversed(type_color.items())):
                ly = min_y + i * legend_vertical_spacing
                lines.append(
                    f'  legend_{t} [shape=box, style=filled, fillcolor="{color}", '
                    f'label="{t}", fontsize=8, fixedsize=false, margin="0.05,0.02", pos="{lx:.3f},{ly:.3f}!"];'
                )
        for a in self.component.agents:
            color = type_color[a.type]
            x, y = pos[id(a)]
            lines.append(f'  a{id(a)} [fillcolor="{color}", pos="{x:.3f},{y:.3f}!"];')
        for u, v in edges:
            lines.append(f"  a{u} -- a{v};")
        lines.append("}")

        return Source("\n".join(lines), engine="neato")


class Monitor:
    """Records the history of the values of observables in a system."""

    system: "System"
    history: dict[str, list[Optional[float]]]  #: Maps observable names to their history

    def __init__(self, system: "System"):
        self.system = system
        self.history = {"time": []} | {obs_name: [] for obs_name in system.observables}

    def __len__(self) -> int:
        """The number of records."""
        return len(self.history["time"])

    def update(self) -> None:
        """Record current time and observable values."""
        self.history["time"].append(self.system.time)
        for obs_name in self.system.observables:
            if obs_name not in self.history:
                self.history[obs_name] = [None] * (len(self.history["time"]) - 1)
            self.history[obs_name].append(self.system[obs_name])

    def measure(self, observable_name: str, time: Optional[float] = None):
        """Get the value of an observable at a specific time.

        Raises:
            AssertionError: If simulation hasn't reached the specified time.
        """
        import bisect

        times: list[int] = list(self.history["time"])
        if time is None:
            time = times[-1]
        assert time <= max(times), "Simulation hasn't reached time {time}"

        return self.history[observable_name][bisect.bisect_right(times, time) - 1]

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the history of observable values as a pandas DataFrame.

        Returns:
            DataFrame with time and observable columns.
        """
        return pd.DataFrame(self.history)

    def tail_mean(
        self,
        observable_name: str,
        tail_fraction: float = 0.1,
    ) -> float:
        """
        Calculate the average value of an observable over a fraction of the tail.

        Args:
            observable_name: Name of the observable to measure.
            tail_fraction: Fraction of the history to consider (from the end).

        Raises:
            AssertionError: If there are not enough measurements.
        """
        window_len = int(tail_fraction * len(self))
        assert (
            len(self) >= window_len and window_len >= 1
        ), f"Not enough measurements ({len(self)}) to calculate tail mean for {observable_name}"

        values = np.asarray(self.history[observable_name][-window_len:], dtype=float)
        return float(np.mean(values))

    def equilibrated(
        self,
        observable_name: Optional[str] = None,
        **equilibration_kwargs,
    ) -> bool:
        """
        Check if an observable (or all observables) has equilibrated based on
        whether the slope of recent values is sufficiently small relative to the mean.

        Args:
            observable_name: Name of the observable to check. If None, checks all observables.
            tail_fraction: Fraction of the history to consider.
            tolerance: Maximum allowed fraction slope deviation from the mean.
        """
        if observable_name is None:
            return all(
                self.equilibrated(obs_name, **equilibration_kwargs)
                for obs_name in self.system.observables
            )

        values = self.history[observable_name]
        times = self.history["time"]
        assert all(v is not None for v in values)
        assert all(t is not None for t in times)
        return equilibrated(values=values, times=times, **equilibration_kwargs)

    def plot(self, combined: bool = False) -> matplotlib.figure.Figure:
        """Make a plot of all observables over time.

        Args:
            combined: Whether to plot all observables on the same axes.
        """
        if combined:
            fig, ax = plt.subplots()
            for obs_name in self.system.observables:
                ax.plot(self.history["time"], self.history[obs_name], label=obs_name)
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Observable")
            plt.margins(0, 0)
        else:
            fig, axs = plt.subplots(
                len(self.system.observables), 1, sharex=True, layout="constrained"
            )
            if len(self.system.observables) == 1:
                axs = [axs]
            for i, obs_name in enumerate(self.system.observables):
                axs[i].plot(self.history["time"], self.history[obs_name], color="black")
                axs[i].set_title(obs_name)
                if i == len(self.system.observables) - 1:
                    axs[i].set_xlabel("Time")
        return fig


def relative_slope(
    values: list[float], times: Optional[list[float]] = None, tail_fraction: float = 0.1
) -> float:
    """
    Computes the magnitude of the slope of the tail of the series.
    Time can be provided to account for non-uniform sampling intervals.

    Raises:
        AssertionError: If there are not enough measurements to compute the slope.
    """
    times = times if times is not None else list(range(len(values)))

    t_tail = times[-1] - tail_fraction * (times[-1] - times[0])
    tail_indices = [i for i, t in enumerate(times) if t >= t_tail]

    assert (
        len(tail_indices) >= 2
    ), f"Not enough measurements ({len(tail_indices)}) to compute slope"

    tail_times = [times[i] for i in tail_indices]
    tail_values = [values[i] for i in tail_indices]
    slope, _ = np.polyfit(tail_times, tail_values, deg=1)

    return float(slope / np.mean(tail_values))


def equilibrated(
    values: list[float],
    times: Optional[list[float]] = None,
    tail_fraction: float = 0.1,
    tolerance: float = 0.01,
) -> bool:
    """
    Checks whether the magnitude of the slope of the tail of the series relative to the mean
    is sufficiently small (below tolerance). Time can be provided to account for non-uniform
    sampling intervals.
    """
    return abs(relative_slope(values, times, tail_fraction)) <= tolerance


def equilibration_time(
    values: list[float],
    times: Optional[list[float]] = None,
    min_tail_length: int = 2,
    tolerance: float = 0.01,
) -> float:
    """Earliest time from which the remaining series is equilibrated."""
    times = times if times is not None else list(range(len(values)))
    for i in range(len(values) - min_tail_length + 1):
        if abs(relative_slope(values[i:], times[i:], tail_fraction=1.0)) <= tolerance:
            return times[i]
    raise ValueError(f"Equilibrium not detected (tolerance={tolerance})")


def equilibrium_value(
    values: list[float],
    times: Optional[list[float]] = None,
    min_tail_length: int = 2,
    tolerance: float = 0.01,
) -> float:
    """Mean of the series from the equilibration point onward."""
    times = times if times is not None else list(range(len(values)))
    eq_time = equilibration_time(values, times, min_tail_length, tolerance)
    eq_index = next(i for i, t in enumerate(times) if t >= eq_time)
    return float(np.mean(values[eq_index:]))


def binding_kinetics_table(system, volume: float = 1.0) -> str:
    """Summarize kinetic constants of two-component binding/unbinding rules
    given volume in liters.
    """

    from pykappa.rule import AVOGADRO
    from pykappa._utils import str_table

    header = ["name", "rule", "k_on", "k_off", "K_D"]
    rows = []

    for fwd_name, rev_name in system._reversible_rules:
        fwd = system.rules[fwd_name]
        rev = system.rules[rev_name]

        fwd_mol = len(fwd.left.components)
        rev_mol = len(rev.left.components)

        if (fwd_mol == 2 and rev_mol == 1) or (fwd_mol == 1 and rev_mol == 2):
            # Determine binding and unbinding reactions
            is_fwd_binding = fwd_mol == 2
            binding_rxn = fwd if is_fwd_binding else rev
            unbinding_rxn = rev if is_fwd_binding else fwd

            binding_types = sorted(
                comp.agents[0].type for comp in binding_rxn.left.components
            )
            unbinding_types = sorted(
                agent.type
                for comp in unbinding_rxn.left.components
                for agent in comp.agents
                if agent is not None
            )
            if binding_types == unbinding_types:
                k_on = binding_rxn.rate(system) * AVOGADRO * volume
                k_off = unbinding_rxn.rate(system)
                kd = k_off / k_on
                rows.append(
                    [
                        f"{fwd_name}/{rev_name}",
                        f"{fwd.left.kappa_str} <-> {fwd.right.kappa_str}",
                        f"{k_on:.2e}",
                        f"{k_off:.2e}",
                        f"{kd:.2e}",
                    ]
                )

    return str_table(rows, header)


def contact_map(system: "System"):
    assert shutil.which("KaSa"), "KaSa not found in the PATH."

    with tempfile.TemporaryDirectory() as tmpdir:
        inp = os.path.join(tmpdir, "in.ka")
        with open(inp, "w") as f:
            f.write(system.kappa_str)

        os.system(
            f"KaSa {inp} --reset-all --compute-contact-map "
            f"--output-directory {tmpdir} "
            f"--output-contact-map out"
        )

        with open(os.path.join(tmpdir, "out.dot")) as f:
            dot = f.read()

    return Source(dot, engine="neato")
