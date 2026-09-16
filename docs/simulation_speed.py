#!/usr/bin/env python3
"""Benchmark PyKappa and KaSim on small rule-based models.

Run this manually, for example:

    python docs/simulation_speed.py
"""

import csv
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, pstdev

from pykappa import System

RULESETS = [
    (
        "Heterodimerization",
        """
        %init: {n} A()
        %init: {n} B()
        %obs: 'AB' |A(x[1]), B(x[1])|
        A(x[.]), B(x[.]) <-> A(x[1]), B(x[1]) @ 1, 1
        """,
    ),
    (
        "Tiling",
        """
        %init: {n} A()
        A(l[.]), A(r[.]) <-> A(l[1]), A(r[1]) @ 1, 1
        A(u[.]), A(d[.]) <-> A(u[1]), A(d[1]) @ 1, 1
        """,
    ),
    (
        "Cyclization",
        """
        %init: {n} A()
        A(r[.]), A(l[.]) <-> A(r[1]), A(l[1]) @ 1 {{1}}, 1
        """,
    ),
]
ENGINES = ("PyKappa", "KaSim")
N_EVENTS = 1000
N_RUNS = 10
AGENT_COUNTS = [round(10 ** (1 + exponent * 0.5)) for exponent in range(9)]
OUTPUT_PATH = Path(__file__).parent / "source/examples/simulation_speed.csv"


def run_once(job: tuple[int, int, int, str]) -> float:
    ruleset_id, n_agents, seed, engine = job
    _, ka_code = RULESETS[ruleset_id]
    ka_code = ka_code.format(n=n_agents)
    start = time.perf_counter()

    if engine == "KaSim":
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "model.ka"
            input_path.write_text(ka_code)
            subprocess.run(
                [
                    "KaSim",
                    str(input_path),
                    "-u",
                    "event",
                    "-l",
                    str(N_EVENTS),
                    "-seed",
                    str(seed),
                    "-d",
                    temporary_directory,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
    else:
        system = System.from_ka(ka_code, seed=seed)
        while system.tally_totals.applied < N_EVENTS:
            system.update()

    return time.perf_counter() - start


if __name__ == "__main__":
    jobs = [
        (ruleset_id, n_agents, seed, engine)
        for engine in ENGINES
        for ruleset_id in range(len(RULESETS))
        for n_agents in AGENT_COUNTS
        for seed in range(N_RUNS)
    ]

    with ProcessPoolExecutor() as executor:
        timings = list(executor.map(run_once, jobs))

    grouped: defaultdict[tuple[str, int, int], list[float]] = defaultdict(list)
    for (ruleset_id, n_agents, _, engine), timing in zip(jobs, timings):
        grouped.setdefault((engine, ruleset_id, n_agents), []).append(timing)

    rows = [
        {
            "engine": engine,
            "ruleset_id": ruleset_id,
            "ruleset": ruleset,
            "agents_per_species": n_agents,
            "n_runs": N_RUNS,
            "n_events": N_EVENTS,
            "mean_wall_time_s": mean(grouped[(engine, ruleset_id, n_agents)]),
            "std_wall_time_s": pstdev(grouped[(engine, ruleset_id, n_agents)]),
        }
        for engine in ENGINES
        for ruleset_id, (ruleset, _) in enumerate(RULESETS)
        for n_agents in AGENT_COUNTS
    ]

    with OUTPUT_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
