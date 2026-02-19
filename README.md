<div align="center">
  <img src="https://raw.githubusercontent.com/berkalpay/pykappa/main/docs/source/_static/logo.png" width="100">
</div>

# PyKappa

[![PyPI](https://img.shields.io/pypi/v/pykappa)](https://pypi.org/project/pykappa)

PyKappa is a Python package for working with rule-based models.
It supports simulation and analysis of a wide variety of systems whose individual components interact as described by rules that transform these components in specified ways and at specified rates.
See our website [pykappa.org](https://pykappa.org) for a tutorial, examples, and documentation.


## Development

Developer requirements can be installed via:
```
pip install -e ".[dev]"
```

To run correctness tests, run `pytest`.
Running `./tests/cpu-profiles/run_profiler.sh` will CPU-profile predefined Kappa models and write the results to `tests/cpu-profiles/results`.
We use the Black code formatter, which can be run as `black .`



