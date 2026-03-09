import math
import colorsys
from typing import TYPE_CHECKING, Optional

from graphviz import Source
import numpy as np

if TYPE_CHECKING:
    from pykappa.pattern import Component


class ComponentPlot:
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
