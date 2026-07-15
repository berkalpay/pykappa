<div align="center">
  <img src="https://raw.githubusercontent.com/berkalpay/pykappa/main/docs/source/_static/logo.svg" width="100">
</div>

<h1 align="center">PyKappa</h1>

[![PyPI](https://img.shields.io/pypi/v/pykappa)](https://pypi.org/project/pykappa)

PyKappa is a Python package for simulation and analysis of rule-based models, which describe systems in terms of local, stochastic graph transformations.
See our website [pykappa.org](https://pykappa.org) for examples and documentation.


## Development

Developer requirements can be installed via `pip install -e ".[dev]"`.
Correctness tests are run via `pytest`.
Running `./tests/cpu-profiles/run_profiler.sh` will CPU-profile predefined Kappa models and write the results to `tests/cpu-profiles/results`.
We use the Black code formatter, which can be run as `black .`
