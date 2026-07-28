from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
RESULTS_DIR = SCRIPT_PATH.with_name("results")


def heterodimerization_small() -> None:
    tests_dir = SCRIPT_PATH.parent.parent
    sys.path.insert(0, str(tests_dir))
    from test_system import heterodimerization_system

    system = heterodimerization_system()
    while system.time < 2:
        system.update()


SYSTEM_CASES = {
    "ktam": (
        [
            "A(l[.]), A(r[.]) <-> A(l[1]), A(r[1]) @ 25.0, 25.0",
            "A(u[.]), A(d[.]) <-> A(u[1]), A(d[1]) @ 25.0, 25.0",
        ],
        1,
    ),
    "uni_bi_small": (
        [
            "A(l[.]), A(r[.]) <-> A(l[1]), A(r[1]) @ 25.0 {25.0}, 25.0",
            "A(u[.]), A(d[.]) <-> A(u[1]), A(d[1]) @ 25.0 {25.0}, 25.0",
        ],
        1,
    ),
}
CASES = ("heterodimerization_small", *SYSTEM_CASES)


def run_case(case: str) -> None:
    if case == "heterodimerization_small":
        heterodimerization_small()
        return

    from pykappa import System

    rules, end_time = SYSTEM_CASES[case]
    system = System.from_kappa({"A(l[.], r[.], u[.], d[.])": 200}, rules)
    while system.time < end_time:
        system.update()


def peak_memory(memprofile: Path) -> float | None:
    values = (
        float(line.split()[1])
        for line in memprofile.read_text().splitlines()
        if line.startswith("MEM") and len(line.split()) >= 3
    )
    return max(values, default=None)


def write_summary(case: str, runtime: float, memory: float | None) -> None:
    summary_path = RESULTS_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    profile_name = f"profile_{case}"
    summary[profile_name] = {
        "timestamp": time.time(),
        "runtime (s)": runtime,
        "peak_memory (MB)": memory,
        "flamegraph": f"{profile_name}_flamegraph.svg",
        "memplot": f"{profile_name}_memplot.png",
        "memprofile": f"{profile_name}_memprofile.dat",
    }
    summary_path.write_text(json.dumps(summary, indent=4) + "\n")


def profile_case(case: str, commit_hash: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    profile_name = f"profile_{case}"
    memprofile = RESULTS_DIR / f"{profile_name}_memprofile.dat"
    memplot = RESULTS_DIR / f"{profile_name}_memplot.png"
    flamegraph = RESULTS_DIR / f"{profile_name}_flamegraph.svg"
    case_command = [sys.executable, str(SCRIPT_PATH), "--run", case]

    print(f"Profiling {case} (commit {commit_hash})")
    subprocess.run(["mprof", "run", "-o", str(memprofile), *case_command], check=True)
    subprocess.run(
        [
            "mprof",
            "plot",
            "-o",
            str(memplot),
            "-t",
            f"{profile_name}, commit {commit_hash}",
            str(memprofile),
        ],
        check=True,
    )

    start = time.perf_counter()
    subprocess.run(
        [
            "py-spy",
            "record",
            "-o",
            str(flamegraph),
            "--",
            *case_command,
        ],
        check=True,
    )
    write_summary(case, time.perf_counter() - start, peak_memory(memprofile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=CASES, help="run one profiling case")
    parser.add_argument("cases", nargs="*", choices=CASES, help="cases to profile")
    args = parser.parse_args()
    if args.run:
        run_case(args.run)
        sys.exit()

    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    for case in args.cases or CASES:
        profile_case(case, commit_hash)
